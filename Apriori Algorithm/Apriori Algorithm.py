# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Open and read the CSV file to determine the number of columns
with open ("C:\\Users\\User\\Desktop\\Vilnius\\data mining\\groceries.csv", 'r') as temp_f:
    col_count = [len(l.split(",")) for l in temp_f.readlines()]

# Close the file
del temp_f 

# Create column names based on the maximum number of columns found in the file
column_names = [i for i in range(0, max(col_count))]

try:
    # Read the CSV file again with specified column names and other settings
    dataset = pd.read_csv("C:\\Users\\User\\Desktop\\Vilnius\\data mining\\groceries.csv", header=None, delimiter=",", names=column_names)
except pd.errors.ParserError as e:
    # Handle any errors that may occur while reading the CSV file
    print(f"Error reading CSV file: {e}")

# Convert the dataset to a list of transactions
transactions = dataset.values.astype(str).tolist()

# Remove 'nan' values from each transaction
transactions = [[item for item in row if item != 'nan'] for row in transactions]

# Create a TransactionEncoder to prepare the data for Apriori algorithm
te = TransactionEncoder() # i do it to trasform list of transactions into a binary format (association rule)
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)# each column corresponds to an item, and each row represents a transaction

# Display the first 5 rows of the transformed DataFrame
df.head(5)

# Get the shape of the DataFrame
df.shape

# Set the minimum support threshold for frequent itemsets
min_support = 0.01

# Generate frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

# Display the top 15 frequent itemsets
print(f"Top {frequent_itemsets.shape[0]} frequent itemsets with minimum support of {min_support}:\n")
print(frequent_itemsets.nlargest(n=15, columns='support'))

# Create a visualization of the top 15 frequent itemsets
plt.figure(figsize=(12, 6))
plt.xticks(rotation=90)
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Top 15 Frequent Itemsets')
sns.barplot(x='itemsets', y='support', data=frequent_itemsets.nlargest(n=15, columns='support'))
plt.show()

# Set the minimum lift threshold for association rules
min_lift = 1.0

# Generate association rules based on frequent itemsets and lift threshold
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

# Display association rules
print(f"\nAssociation rules with minimum lift of {min_lift}:\n")
print(rules.sort_values(by=['support'], ascending=False))

# Calculate the length of the antecedent for each rule
rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))

# Set minimum antecedent length and minimum confidence for filtering rules
min_antecedent_len = 2
min_confidence = 0.3

# Filter rules based on antecedent length and confidence
filtered_rules = rules[
    (rules['antecedent_len'] >= min_antecedent_len) &
    (rules['confidence'] > min_confidence)
]

# Display filtered rules
print(f"\nFiltered rules (antecedent length >= {min_antecedent_len}, confidence > {min_confidence}):\n")
print(filtered_rules.sort_values(by=['lift', 'support'], ascending=False))
