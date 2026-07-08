import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("data.csv")

# Total Sales Column
df["Total Sales"] = df["Quantity"] * df["Price"]

print("\nRetail Sales Dataset")
print(df)

print("\nTotal Revenue")
print(df["Total Sales"].sum())

print("\nSales by Product")
print(df.groupby("Product")["Total Sales"].sum())

print("\nSales by City")
print(df.groupby("City")["Total Sales"].sum())

# Monthly Sales
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

monthly_sales = df.groupby("Month")["Total Sales"].sum()

# Chart 1
plt.figure(figsize=(8,5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

# Chart 2
product_sales = df.groupby("Product")["Total Sales"].sum()

plt.figure(figsize=(6,5))
product_sales.plot(kind="bar")
plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.show()

# Chart 3
city_sales = df.groupby("City")["Total Sales"].sum()

plt.figure(figsize=(6,6))
city_sales.plot(kind="pie", autopct="%1.1f%%")
plt.title("City-wise Sales")
plt.ylabel("")
plt.show()