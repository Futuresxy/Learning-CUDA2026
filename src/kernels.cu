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
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {      
    
    // [Config] 精度与常量
    using AccT = float; // 累加器必须用 float
    const AccT NEG_INF = -std::numeric_limits<AccT>::infinity();
    const int Br = 32; 
    const int Bc = 32; 
    const AccT softmax_scale = 1.0f / std::sqrt(static_cast<AccT>(head_dim));

    // [Fix 2] 使用 size_t 防止大 Tensor 索引溢出 (RTX 4090 显存较大，int 容易溢出)
    size_t q_head_stride = head_dim;
    size_t q_seq_stride = static_cast<size_t>(query_heads) * head_dim;
    size_t q_batch_stride = static_cast<size_t>(target_seq_len) * q_seq_stride;

    size_t k_head_stride = head_dim;
    size_t k_seq_stride = static_cast<size_t>(kv_heads) * head_dim;
    size_t k_batch_stride = static_cast<size_t>(src_seq_len) * k_seq_stride;
    
    // V 的 stride 通常与 K 一致
    size_t v_head_stride = head_dim;
    size_t v_seq_stride = static_cast<size_t>(kv_heads) * head_dim;
    size_t v_batch_stride = static_cast<size_t>(src_seq_len) * v_seq_stride;

    size_t o_head_stride = head_dim;
    size_t o_seq_stride = static_cast<size_t>(query_heads) * head_dim;
    size_t o_batch_stride = static_cast<size_t>(target_seq_len) * o_seq_stride;

    // [Fix 1] 计算 GQA 分组大小。如果是 MHA，group_size 为 1。
    // 标准 GQA 映射是块状映射 (0,0,1,1) 而不是循环映射 (0,1,0,1)
    int group_size = query_heads / kv_heads; 

    for(int b = 0; b < batch_size; ++b) {
        for(int th = 0; th < query_heads; ++th) {
            
            // [Fix 1] GQA Mapping: 使用除法而不是取模
            int kh = th / group_size; 

            // 预计算当前 batch 和 head 的基地址
            const T* q_base_ptr = h_q.data() + b * q_batch_stride + th * q_head_stride; 
            const T* k_base_ptr = h_k.data() + b * k_batch_stride + kh * k_head_stride;
            const T* v_base_ptr = h_v.data() + b * v_batch_stride + kh * v_head_stride;
            T* o_base_ptr = h_o.data() + b * o_batch_stride + th * o_head_stride;

            // --- Outer Loop: Query Blocks (Row) ---
            for(int t_block = 0; t_block < (target_seq_len + Br - 1) / Br; ++t_block) {
                
                // 初始化 FlashAttention 统计量
                std::vector<std::vector<AccT>> O_block(Br, std::vector<AccT>(head_dim, 0.0f)); 
                std::vector<AccT> l_block(Br, 0.0f);          
                std::vector<AccT> m_block(Br, NEG_INF);       

                // --- Inner Loop: Key/Value Blocks (Column) ---
                for(int s_block = 0; s_block < (src_seq_len + Bc - 1) / Bc; ++s_block) {
                    
                    // 每一个 Query Block 内的行
                    for(int tr = 0; tr < Br; ++tr) {
                        int t_index = t_block * Br + tr; 
                        if(t_index >= target_seq_len) continue;

                        // 状态变量
                        AccT m_prev = m_block[tr]; 
                        AccT m_curr = NEG_INF; 
                        
                        // 暂存 Score，避免重复计算
                        std::vector<AccT> S_row(Bc, NEG_INF); 
                        const T* q_ptr_row = q_base_ptr + t_index * q_seq_stride;

                        // [Step 1] 计算 Q * K^T + Masking
                        for(int sc = 0; sc < Bc; ++sc) {
                            int s_index = s_block * Bc + sc;
                            
                            // 检查 Mask 条件 (Padding Mask 或 Causal Mask)
                            bool masked = false;
                            if (s_index >= src_seq_len) masked = true;
                            if (is_causal && s_index > t_index) masked = true;

                            if (masked) {
                                S_row[sc] = NEG_INF;
                                continue;
                            }

                            // 计算点积
                            const T* k_ptr_row = k_base_ptr + s_index * k_seq_stride;
                            AccT score = 0.0f;
                            for(int d = 0; d < head_dim; ++d) {
                                score += static_cast<AccT>(q_ptr_row[d]) * static_cast<AccT>(k_ptr_row[d]);
                            }
                            score *= softmax_scale;
                            
                            S_row[sc] = score;
                            if (score > m_curr) m_curr = score;
                        }

                        // 如果当前块全是 Mask (m_curr 仍为 -inf)，则跳过更新
                        if (m_curr == NEG_INF) continue;

                        // [Step 2] Online Softmax 更新
                        AccT m_new = std::max(m_prev, m_curr);
                        
                        // 计算缩放因子 rescale = exp(m_prev - m_new)
                        // 特殊处理: 如果 m_prev 是 -inf (第一次有效更新)，则 rescale 设为 0 (丢弃之前的 0 值)
                        AccT rescale = (m_prev == NEG_INF) ? 0.0f : std::exp(m_prev - m_new);

                        // 计算当前块的 P (unnormalized probs) 和 l_curr
                        AccT l_curr = 0.0f;
                        std::vector<AccT> P_row(Bc, 0.0f);
                        for(int sc = 0; sc < Bc; ++sc) {
                            if (S_row[sc] != NEG_INF) {
                                P_row[sc] = std::exp(S_row[sc] - m_new);
                                l_curr += P_row[sc];
                            }
                        }

                        // [Step 3] 更新 Output 和 l
                        // O_new = O_prev * rescale + P_curr * V_curr
                        
                        // 3.1 Rescale 旧的 O
                        // 优化：如果 rescale 为 0 (首次更新)，直接清零 O_block (虽然初始化是0，但逻辑上更安全)
                        if (rescale == 0.0f) {
                             // Do nothing, O_block is assumed 0 or will be overwritten if we didn't accumulate
                             // 实际上这里乘 0 即可
                             for(int d = 0; d < head_dim; ++d) O_block[tr][d] = 0.0f; 
                        } else {
                             for(int d = 0; d < head_dim; ++d) O_block[tr][d] *= rescale;
                        }

                        // 3.2 累加新的 P * V
                        for(int sc = 0; sc < Bc; ++sc) {
                            // 只有概率不为 0 才计算 (稀疏加速)
                            if (P_row[sc] == 0.0f) continue;

                            int s_index = s_block * Bc + sc;
                            const T* v_ptr_row = v_base_ptr + s_index * v_seq_stride;
                            
                            AccT p_val = P_row[sc];
                            for(int d = 0; d < head_dim; ++d) {
                                O_block[tr][d] += p_val * static_cast<AccT>(v_ptr_row[d]);
                            }
                        }

                        // 3.3 更新 l 和 m
                        l_block[tr] = l_block[tr] * rescale + l_curr;
                        m_block[tr] = m_new;

                    } // End tr
                } // End s_block

                // [Step 4] 最终归一化并写入 Output
                for(int tr = 0; tr < Br; ++tr) {
                    int t_index = t_block * Br + tr;
                    if(t_index >= target_seq_len) continue;

                    T* o_ptr_row = o_base_ptr + t_index * o_seq_stride;
                    
                    AccT divisor = l_block[tr];
                    // 防止全 Mask 导致的除零 (输出 0)
                    if (divisor < 1e-6f) {
                        for(int d = 0; d < head_dim; ++d) o_ptr_row[d] = T(0);
                    } else {
                        AccT inv_divisor = 1.0f / divisor;
                        for(int d = 0; d < head_dim; ++d) {
                            o_ptr_row[d] = static_cast<T>(O_block[tr][d] * inv_divisor);
                        }
                    }
                }
            } // End t_block
        } // End th
    } // End batch
}

// template <typename T>
// void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
//                     const std::vector<T> &h_v, std::vector<T> &h_o,
//                     int batch_size, int target_seq_len, int src_seq_len,
//                     int query_heads, int kv_heads, int head_dim, bool is_causal)
// {
//   // TODO: Implement the flash attention function
//   // h_q shape: [batch_size, target_seq_len, query_heads, head_dim]
//   // h_k shape: [batch_size, src_seq_len, kv_heads, head_dim]
//   // h_v shape: [batch_size, src_seq_len, kv_heads, head_dim]
//   // h_o shape: [batch_size, target_seq_len, query_heads, head_dim]
//   const int Br = 16; // Block size for target sequence length
//   const int Bc = 16; // Block size for source sequence length
//   // divide the h_q to [target_seq_len/Br] Bolcks shape:   [Batch_size , Br,  query_heads,  head_dim]
//   // divide the h_k, h_v to [src_seq_len/Bc] Blocks shape: [Batch_size , Bc,  kv_heads,  head_dim]

//   for (int b = 0; b < batch_size; ++b)
//   {
//     for (int th = 0; th < query_heads; ++th)
//     {
//       for (int t_block = 0; t_block < (target_seq_len + Br - 1) / Br; ++t_block)
//       { // Query blocks
//         // Initialize accumulators
//         vector<vector<T>> dot_product(Br, vector<T>(Bc, T(0)));  // Store dot products also Psum  for the block
//         vector<T> sum_data(Br, T(0));                            // Store sum of exponentials for softmax normalization
//         vector<T> max_data(Br, T(-1e9));                         // Store max values for numerical stability
//         vector<vector<T>> Output(Br, vector<T>(head_dim, T(0))); // Store sum of exponentials for softmax normalization

//         for (int s_block = 0; s_block < (src_seq_len + Bc - 1) / Bc; ++s_block)
//         { // Key Value blocks

//           // Process each Query block processing
//           for (int tr = 0; tr < Br; ++tr)
//           {
//             int t_index = t_block * Br + tr; // Actual target sequence index
//             if (t_index >= target_seq_len)
//               continue;

//             // K V block processing
//             // Temporary storage for dot products sum and max
//             T max_val = max_data[tr];
//             T block_sum = T(0);
//             for (int sc = 0; sc < Bc; ++sc)
//             { // one target token in Qi compute Qii * Kj
//               int s_index = s_block * Bc + sc;
//               if (s_index >= src_seq_len)
//                 continue;
//               // Compute attention score and output
//               for (int d = 0; d < head_dim; ++d)
//               {
//                 T q_val = h_q[((b * target_seq_len + t_index) * query_heads + th) * head_dim + d];
//                 T k_val = h_k[((b * src_seq_len + s_index) * kv_heads + (th % kv_heads)) * head_dim + d];
//                 dot_product[tr][sc] += q_val * k_val;
//               }

//               // Apply softmax normalization and causal masking if needed
//               if (is_causal && s_index > t_index)
//               {
//                 dot_product[tr][sc] = T(-1e9); // Mask future positions
//               }
//               // find max for numerical stability
//               max_val = dot_product[tr][sc] > max_val ? dot_product[tr][sc] : max_val;
//             }
//             for (int sc = 0; sc < Bc; ++sc)
//             {
//               dot_product[tr][sc] = std::exp(dot_product[tr][sc] - max_val); // Exponential for softmax
//               block_sum += dot_product[tr][sc];
//             }

//             // Update sum and max for the block
//             sum_data[tr] = sum_data[tr] * std::exp(max_data[tr] - max_val) + block_sum;

//             // compute the output for the block

//             for (int d = 0; d < head_dim; ++d)
//             {
//               Output[tr][d] *= std::exp(max_data[tr] - max_val); // Update output with previous max values
//               for (size_t i = 0; i < Bc; i++)
//               {
//                 Output[tr][d] += dot_product[tr][i] * h_v[((b * src_seq_len + i) * kv_heads + (th % kv_heads)) * head_dim + d];
//               }
//             }
//             max_data[tr] = max_val;
//           }
//         }
//         // Normalize the output by the sum of exponentials
//         for (int tr = 0; tr < Br; ++tr)
//         {
//           int t_index = t_block * Br + tr; // Actual target sequence index
//           if (t_index >= target_seq_len)
//             continue;
//           for (int d = 0; d < head_dim; ++d)
//           {
//             h_o[((b * target_seq_len + t_index) * query_heads + th) * head_dim + d] = Output[tr][d] / sum_data[tr];
//           }
//         }
//       }
//     }
//   }

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
