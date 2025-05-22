from PIL import Image
import numpy as np
import soundfile as sf

# Параметры
sampling_rate = 22050
duration = 0.01
image_width = 100
image_height = 100

# Чтение аудиофайла
audio_data, sr = sf.read('output_rgb.wav')
assert sr == sampling_rate, "Частота дискретизации не совпадает!"

# Вычисляем количество выборок на канал
samples_per_channel = int(sampling_rate * duration)
num_pixels = image_width * image_height
num_channels = 3  # R, G, B

# Проверяем длину аудиофайла
assert len(audio_data) >= num_pixels * num_channels * samples_per_channel, "Аудиофайл слишком короткий!"

# Массив для восстановленных каналов
rgb_values = []

# Обрабатываем аудиоданные
for i in range(num_pixels * num_channels):
    start = i * samples_per_channel
    end = start + samples_per_channel
    segment = audio_data[start:end]
    amplitude = np.sqrt(np.mean(segment**2)) if len(segment) > 0 else 0
    value = min(max(int(amplitude * 255), 0), 255)
    rgb_values.append(value)

# Преобразуем в массив пикселей (height, width, 3)
rgb_values = np.array(rgb_values).reshape(image_height, image_width, num_channels)

# Создаём цветное изображение
restored_image = Image.fromarray(rgb_values.astype(np.uint8), mode='RGB')
restored_image.save('restored_image_rgb.png')
print("Цветное изображение восстановлено и сохранено в 'restored_image_rgb.png'")