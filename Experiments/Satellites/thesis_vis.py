import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

# Пример словаря с наборами данных
data_dict = {
    "A": np.random.normal(10, 5, 100),
    "B": np.random.normal(12, 5, 100),
    "C": np.random.normal(20, 5, 100),
    "D": np.random.normal(10, 10, 100)
}

# Получаем список ключей
keys = list(data_dict.keys())

# Создаем пустую DataFrame для хранения p-values
p_values = pd.DataFrame(np.ones((len(keys), len(keys))), index=keys, columns=keys)

# Заполняем DataFrame результатами теста Стьюдента
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        t_stat, p_val = stats.ttest_ind(data_dict[keys[i]], data_dict[keys[j]], equal_var=False)
        p_values.iloc[i, j] = p_val
        p_values.iloc[j, i] = p_val  # Заполняем симметрично

# Создаем маску для значений p > 0.05 (незначимых)
mask = p_values > 0.05

# Создаем кастомную цветовую карту: серый для p > 0.05, градиент для остальных
cmap = sns.color_palette("Reds", as_cmap=True)
cmap.set_bad(color='lightgray')  # Цвет для незначимых значений

# Рисуем heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(p_values, annot=True, fmt=".3f", cmap=cmap, mask=mask, cbar=True,
            linewidths=0.5, linecolor="black")

plt.title("Heatmap p-values (Student's t-test)")
plt.show()
