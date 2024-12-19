import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 加载 .mat 文件
file_1_1 = 'exp/1-1.mat'
file_1_2 = 'exp/1-2.mat'
file_3 = 'exp/a1.mat'

# 加载数据
data_1_1 = sio.loadmat(file_1_1)['denoised_signal']
data_1_2 = sio.loadmat(file_1_2)['denoised_signal']
mat_data = sio.loadmat(file_3)['mat']

# 提取实部和虚部
real_data_1_1 = np.real(data_1_1)
imag_data_1_1 = np.imag(data_1_1)
real_data_1_2 = np.real(data_1_2)
imag_data_1_2 = np.imag(data_1_2)

# 处理实部和虚部
real_add = np.abs((real_data_1_1 + real_data_1_2) / 2)
real_sub = np.abs((real_data_1_1 - real_data_1_2) / 2)
real_result = np.abs(real_add - real_sub)

imag_add = np.abs((imag_data_1_1 + imag_data_1_2) / 2)
imag_sub = np.abs((imag_data_1_1 - imag_data_1_2) / 2)
imag_result = np.abs(imag_add - imag_sub)

# 合成复数结果
final_result = real_result + 1j * imag_result
final_result_magnitude = np.abs(final_result)

# 对 final_result_magnitude 进行归一化
final_min = final_result_magnitude.min()
final_max = final_result_magnitude.max()
final_result_normalized = (final_result_magnitude - final_min) / (final_max - final_min)

# 可视化归一化的合成结果
plt.figure()
plt.imshow(final_result_normalized, cmap='gray')
plt.title('Final Result (Normalized)')
plt.colorbar()
plt.show()

# 归一化第三个mat数据
mat_magnitude = np.abs(mat_data)
mat_min = mat_magnitude.min()
mat_max = mat_magnitude.max()
mat_normalized = (mat_magnitude - mat_min) / (mat_max - mat_min)



# 比较两个归一化结果
comparison_result_normalized = np.minimum(final_result_normalized, mat_normalized)

# # 反归一化结果
# comparison_result = comparison_result_normalized * (mat_max - mat_min) + mat_min

#可视化做出让步的结果
comparison_result_normalized[comparison_result_normalized > 0.03] = 0.03
mat_normalized[mat_normalized > 0.03] = 0.03


# 可视化第三个mat的原始数据
plt.figure()
plt.imshow(mat_normalized, cmap='gray')
plt.title('Original Mat Data (Before Normalization)')
plt.colorbar()
plt.show()

# 可视化反归一化的比较结果
plt.figure()
plt.imshow(comparison_result_normalized, cmap='gray')
plt.title('Comparison Result (After Denormalization)')
plt.colorbar()
plt.show()
