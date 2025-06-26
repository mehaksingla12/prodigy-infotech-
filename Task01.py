import matplotlib.pyplot as plt

age_groups = ['0‑14', '15‑24', '25‑54', '55‑64', '65+']
percentages = [26.3, 17.5, 41.6, 7.9, 6.7]  # based on indexmundi data

plt.figure(figsize=(8,5))
plt.bar(age_groups, percentages, color='skyblue', edgecolor='black')
plt.title('India Population Age Distribution (~2020)')
plt.xlabel('Age Group (years)')
plt.ylabel('Percentage of Total Population')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
