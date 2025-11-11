import numpy as np
import matplotlib.pyplot as plt
print("Student Study Hour vs Exam Score Prediction using Linear Regression\n")


X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
y = np.array([35, 45, 50, 55, 60, 70, 75, 80, 85], dtype=float)


plt.scatter(X, y, color='blue', label='Data Points')
plt.title('Hours Studied vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.show()


n = len(X)
m = (n * np.sum(X * y) - np.sum(X) * np.sum(y)) / (n * np.sum(X**2) - (np.sum(X))**2)
b = (np.sum(y) - m * np.sum(X)) / n

print(f"Slope (m): {m:.2f}")
print(f"Intercept (b): {b:.2f}")


y_pred = m * X + b


plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression - Study Hours vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.show()


try:
    hours = float(input("\nEnter the number of study hours: "))

    if hours > 10:
        print("A student cannot study more than 10 hours in a day! Please enter a value between 0 and 10.")
    elif hours < 0:
        print("Study hours cannot be negative! Please enter a positive number.")
    else:
        predicted_score = m * hours + b

    
        if predicted_score > 100:
            predicted_score = 100

        print(f"Predicted Score for {hours} hours of study = {predicted_score:.2f}")

except ValueError:
    print("Please enter a valid number for hours.")


ss_total = np.sum((y - np.mean(y))**2)
ss_residual = np.sum((y - y_pred)**2)
r2 = 1 - (ss_residual / ss_total)
print(f"\nModel R² Score: {r2:.4f}")
