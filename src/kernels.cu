#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  size_t min_dim = std::min(rows, cols);// Determine the smaller dimension for non-square matrices
  if(min_dim <= 0) 
    return T(0); // Handle edge case of zero-sized dimension

  T trace_sum = T(0);
  for (size_t i = 0; i < min_dim; ++i) {
    trace_sum += h_input[i * cols + i]; // Access diagonal element
  }
  return trace_sum;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template<typename T>
struct Precision {
    using Type = float;
};

template<>
struct Precision<float> {
    using Type = double; 
};

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {      
    
    // 1. 选择累加器精度 (float->double, half->float)
    using AccT = typename Precision<T>::Type;
    
    const AccT NEG_INF = -std::numeric_limits<AccT>::infinity();
    const int Br = 256; 
    const int Bc = 256; 

    // [关键优化] 缩放因子对齐
    // 先用 float 计算 sqrt，确保与 FP32 模型的定义一致，然后再转为 AccT 进行高精度乘法。
    // 如果直接用 double 计算 sqrt，会导致常数与 Reference 产生 1e-9 的偏差，累积后会导致测试失败。
    const float scale_f = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const AccT softmax_scale = static_cast<AccT>(scale_f);

    // 2. 使用 size_t 防止大 Tensor 索引溢出
    size_t q_head_stride = head_dim;
    size_t q_seq_stride = static_cast<size_t>(query_heads) * head_dim;
    size_t q_batch_stride = static_cast<size_t>(target_seq_len) * q_seq_stride;

    size_t k_head_stride = head_dim;
    size_t k_seq_stride = static_cast<size_t>(kv_heads) * head_dim;
    size_t k_batch_stride = static_cast<size_t>(src_seq_len) * k_seq_stride;
    
    size_t v_head_stride = head_dim;
    size_t v_seq_stride = static_cast<size_t>(kv_heads) * head_dim;
    size_t v_batch_stride = static_cast<size_t>(src_seq_len) * v_seq_stride;

    size_t o_head_stride = head_dim;
    size_t o_seq_stride = static_cast<size_t>(query_heads) * head_dim;
    size_t o_batch_stride = static_cast<size_t>(target_seq_len) * o_seq_stride;

    int group_size = query_heads / kv_heads; 

    for(int b = 0; b < batch_size; ++b) {
        for(int th = 0; th < query_heads; ++th) {
            
            int kh = th / group_size; 

            // 预计算基地址
            const T* q_base_ptr = h_q.data() + b * q_batch_stride + th * q_head_stride; 
            const T* k_base_ptr = h_k.data() + b * k_batch_stride + kh * k_head_stride;
            const T* v_base_ptr = h_v.data() + b * v_batch_stride + kh * v_head_stride;
            T* o_base_ptr = h_o.data() + b * o_batch_stride + th * o_head_stride;

            // --- Query Blocks (Row) ---
            for(int t_block = 0; t_block < (target_seq_len + Br - 1) / Br; ++t_block) {
                
                // 状态变量使用高精度 AccT
                std::vector<std::vector<AccT>> O_block(Br, std::vector<AccT>(head_dim, 0.0)); 
                std::vector<AccT> l_block(Br, 0.0);          
                std::vector<AccT> m_block(Br, NEG_INF);       

                // --- Key/Value Blocks (Column) ---
                for(int s_block = 0; s_block < (src_seq_len + Bc - 1) / Bc; ++s_block) {
                    
                    for(int tr = 0; tr < Br; ++tr) {
                        int t_index = t_block * Br + tr; 
                        if(t_index >= target_seq_len) continue;

                        AccT m_prev = m_block[tr]; 
                        AccT m_curr = NEG_INF; 
                        
                        std::vector<AccT> S_row(Bc, NEG_INF); 
                        const T* q_ptr_row = q_base_ptr + t_index * q_seq_stride;

                        // [Step 1] 计算 Q * K^T
                        for(int sc = 0; sc < Bc; ++sc) {
                            int s_index = s_block * Bc + sc;
                            
                            // Mask 逻辑
                            bool masked = false;
                            if (s_index >= src_seq_len) masked = true;//按照分块原子超出src 序列长度部分直接mask
                            if (is_causal && s_index > t_index) masked = true;  //因果掩码 并且s_index 大于 t_index的上三角阵上方部分掩码

                            if (masked) continue; 

                            const T* k_ptr_row = k_base_ptr + s_index * k_seq_stride;
                            AccT score = 0.0;
                            
                            // 强制先转为 AccT (double) 再乘，保证点积精度
                            for(int d = 0; d < head_dim; ++d) {
                                score += static_cast<AccT>(q_ptr_row[d]) * static_cast<AccT>(k_ptr_row[d]);
                            }
                            score *= softmax_scale;
                            
                            S_row[sc] = score;
                            if (score > m_curr) m_curr = score;
                        }

                        if (m_curr == NEG_INF) continue; 

                        // [Step 2] Online Softmax Update
                        AccT m_new = std::max(m_prev, m_curr);
                        
                        // 计算 rescale。如果 m_prev 是 -inf，rescale 为 0
                        AccT rescale = (m_prev == NEG_INF) ? 0.0 : std::exp(m_prev - m_new);

                        AccT l_curr = 0.0;
                        std::vector<AccT> P_row(Bc, 0.0);
                        
                        // 计算 exp 和 当前块的 sum
                        for(int sc = 0; sc < Bc; ++sc) {
                            if (S_row[sc] != NEG_INF) {
                                P_row[sc] = std::exp(S_row[sc] - m_new);
                                l_curr += P_row[sc];
                            }
                        }

                        // [Step 3] 更新 O 和 l
                        
                        // 3.1 Rescale 输出O 更新saftsoftmax 最大值偏移 (旧值缩放)
                        if (rescale != 1.0) {
                             if (rescale == 0.0) {
                                 // 第一次有效更新，清空 O 
                                 std::fill(O_block[tr].begin(), O_block[tr].end(), 0.0);
                             } else {
                                 for(int d = 0; d < head_dim; ++d) O_block[tr][d] *= rescale;
                             }
                        }

                        // 3.2 Accumulate P * V (新值累加)
                        for(int sc = 0; sc < Bc; ++sc) {
                            if (P_row[sc] == 0.0) continue;

                            int s_index = s_block * Bc + sc;
                            const T* v_ptr_row = v_base_ptr + s_index * v_seq_stride;
                            
                            AccT p_val = P_row[sc];
                            for(int d = 0; d < head_dim; ++d) {
                                O_block[tr][d] += p_val * static_cast<AccT>(v_ptr_row[d]);
                            }
                        }

                        // 3.3 更新统计量
                        l_block[tr] = l_block[tr] * rescale + l_curr;
                        m_block[tr] = m_new;

                    } // End tr
                } // End s_block

                // [Step 4] 最终写入
                for(int tr = 0; tr < Br; ++tr) {
                    int t_index = t_block * Br + tr;
                    if(t_index >= target_seq_len) continue;

                    T* o_ptr_row = o_base_ptr + t_index * o_seq_stride;
                    AccT divisor = l_block[tr];
                    
                    if (divisor < 1e-15) { 
                        for(int d = 0; d < head_dim; ++d) o_ptr_row[d] = static_cast<T>(0);
                    } else {
                        // 使用 AccT (double) 进行除法，最大限度减少最后一步的舍入误差
                        for(int d = 0; d < head_dim; ++d) {
                            o_ptr_row[d] = static_cast<T>(O_block[tr][d] / divisor);
                        }
                    }
                }
            } // End t_block
        } 
    } 
}
  // *********************************************************************
  // Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
  // DO NOT MODIFY THIS SECTION
  // *********************************************************************
  template int trace<int>(const std::vector<int> &, size_t, size_t);
  template float trace<float>(const std::vector<float> &, size_t, size_t);
  template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &,
                                      const std::vector<float> &, std::vector<float> &,
                                      int, int, int, int, int, int, bool);
  template void flashAttention<half>(const std::vector<half> &, const std::vector<half> &,
                                     const std::vector<half> &, std::vector<half> &,
                                     int, int, int, int, int, int, bool);
