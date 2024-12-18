# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ��������� ����
process_noise = 0.01  # ��� ��������
measurement_noise = 0.1  # ��� ���������

# ��������� ������� �������
dt = 1.0  # ��������� ���
A = np.array([[1, dt], [0, 1]])  # ������� �������� ���������
B = np.array([[0.5 * dt**2], [dt]])  # ������� ����������
H = np.array([[1, 0]])  # ������� ���������
Q = np.array([[process_noise**2, 0], [0, process_noise**2]])  # �������������� ������� ���� ��������
R = np.array([[measurement_noise**2]])  # �������������� ������� ���� ���������

# ������������� ������� �������
x = np.array([[0], [0]])  # ��������� ��������� [�������, ��������]
P = np.eye(2)  # ��������� �������������� ������� ������

# ��������� ������ � �����������
num_samples = 100
true_position = np.linspace(0, 10, num_samples)
true_velocity = np.ones(num_samples) * 0.1  # �������� �����

# ���������� ���� � ������
measured_position = true_position + np.random.normal(0, measurement_noise, num_samples)
measured_velocity = true_velocity + np.random.normal(0, process_noise, num_samples)

# ���������� ������
filtered_position = []
filtered_velocity = []

for i in range(num_samples):
    # ������������
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # ���������
    z = np.array([[measured_position[i]]])

    # ���������
    y = z - np.dot(H, x)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y)
    P = P - np.dot(np.dot(K, H), P)

    # ���������� �����������
    filtered_position.append(x[0, 0])
    filtered_velocity.append(x[1, 0])

# ������������ �����������
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(true_position, label='True Position')
plt.plot(measured_position, label='Measured Position')
plt.plot(filtered_position, label='Filtered Position')
plt.legend()
plt.title('Position')

plt.subplot(2, 1, 2)
plt.plot(true_velocity, label='True Velocity')
plt.plot(measured_velocity, label='Measured Velocity')
plt.plot(filtered_velocity, label='Filtered Velocity')
plt.legend()
plt.title('Velocity')

plt.tight_layout()
plt.show()