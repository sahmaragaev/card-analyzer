import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    if 'Category' in data.columns:
        le = LabelEncoder()
        data['Category'] = le.fit_transform(data['Category'])
        global category_mapping
        category_mapping = {index: label for index, label in enumerate(le.classes_)}
    return data

# Function to show category mappings
def show_category_mappings():
    for category, number in category_mapping.items():
        print(f"{category}: {number}")

# Function to show spendings for a specific month
def show_monthly_spendings(data, month):
    monthly_data = data[data['Date'].dt.month == month]
    spendings = monthly_data.groupby('Date')['Transaction Amount'].sum()
    print(spendings)

    # Plotting
    
    plt.figure(figsize=(10, 6))
    spendings.plot(kind='bar')
    plt.title(f'Spendings for Month {month}')
    plt.xlabel('Date')
    plt.ylabel('Transaction Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to show spendings for a specific category
def show_category_spendings(data, category):
    category_data = data[data['Category'] == category]
    spendings = category_data.groupby('Category')['Transaction Amount'].sum()
    print(spendings)

    # Plotting
    plt.figure(figsize=(10, 6))
    spendings.plot(kind='bar')
    plt.title(f'Spendings for Category {category}')
    plt.xlabel('Category')
    plt.ylabel('Transaction Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to show spendings for a specific category in a specific month
def show_category_spendings_by_month(data, category, month):
    category_month_data = data[(data['Category'] == category) & (data['Date'].dt.month == month)]
    spendings = category_month_data.groupby(['Category', 'Date'])['Transaction Amount'].sum()
    print(spendings)

    # Plotting
    plt.figure(figsize=(10, 6))
    spendings.plot(kind='bar')
    plt.title(f'Spendings for Category {category} in Month {month}')
    plt.xlabel('Date')
    plt.ylabel('Transaction Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to predict future spendings for a category
def predict_future_spendings(data, category_code):
    # Using the category_code directly as it's already encoded
    category_data = data[data['Category'] == category_code]

    # If no data available for the category
    if category_data.empty:
        return "No data available for this category"

    # Prepare data for Linear Regression
    X = category_data[['Category']]  # Using only the encoded category for simplicity
    y = category_data['Transaction Amount']
    model = LinearRegression()
    model.fit(X, y)

    # Simple prediction for the same category
    prediction_df = pd.DataFrame([[category_code]], columns=['Category'])
    prediction = model.predict(prediction_df)
    return prediction

# Main application
def main():
    file_path = 'credit_card_transaction_flow.csv'
    data = load_data(file_path)

    while True:
        print("\nSelect an option:")
        print("1. View monthly spendings")
        print("2. View category spendings")
        print("3. View category spendings by month")
        print("4. Predict future spendings for a category")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            month = int(input("Enter month (1-12): "))
            show_monthly_spendings(data, month)
        elif choice == '2':
            show_category_mappings()
            category = int(input("Enter category number: "))
            show_category_spendings(data, category)
        elif choice == '3':
            show_category_mappings()
            category = int(input("Enter category number: "))
            month = int(input("Enter month (1-12): "))
            show_category_spendings_by_month(data, category, month)
        elif choice == '4':
            show_category_mappings()
            category = int(input("Enter category number for prediction: "))
            predictions = predict_future_spendings(data, category)
            print("Predicted future spendings for category {}: {}".format(category, predictions))
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
