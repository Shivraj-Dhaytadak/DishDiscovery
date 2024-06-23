import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Function to read recipes from CSV file
def read_recipes_from_csv(file_path):
    recipes = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            name = row[0]
            ingredients = row[1][2:-2].split("', '")
            instructions = row[3][2:-2].split("', '")
            time = int(row[2])
            cuisine = row[4].split(', ')
            course = row[5]
            recipes.append((name, ingredients, instructions, time, cuisine, course))
    return recipes

# Function to combine ingredients and instructions into a single document
def combine_ingredients_instructions(ingredients, instructions):
    return ' '.join(ingredients) + ' ' + ' '.join(instructions)

# Function to generate recipe vectors using TF-IDF
def generate_recipe_vectors(recipes):
    documents = [combine_ingredients_instructions(recipe[1], recipe[2]) for recipe in recipes]
    vectorizer = TfidfVectorizer()
    recipe_vectors = vectorizer.fit_transform(documents)
    return recipe_vectors, vectorizer

# Function to retrieve top-k recipes based on a query
def retrieve_recipes(query, recipe_vectors, vectorizer, recipes, k=5):
    ingredient, max_time, cuisine, course = query
    query_doc = combine_ingredients_instructions([ingredient], [])
    query_vector = vectorizer.transform([query_doc])
    
    # Filter recipes based on cuisine, time, and course
    filtered_recipes = [recipe for recipe in recipes if cuisine in recipe[4] and recipe[3] <= max_time and ingredient.lower() in ' '.join(recipe[1]).lower() and course.lower() in recipe[5].lower()]
    
    if len(filtered_recipes) == 0:
        print("No recipes found matching the query ingredient, cuisine, and course.")
        return []
    
    # Get indices of filtered recipes
    filtered_indices = [recipes.index(recipe) for recipe in filtered_recipes]
    
    # Calculate similarity for filtered recipes
    filtered_vectors = recipe_vectors[filtered_indices]
    similarities = cosine_similarity(query_vector, filtered_vectors)
    top_indices = np.argsort(similarities[0])[::-1][:k]
    
    top_recipes = [filtered_recipes[i] for i in top_indices]
    
    return top_recipes

start = time.time()

# Read recipes from CSV file
recipes = read_recipes_from_csv('recipes2.csv')  # Update file path to your CSV file

# Generate recipe vectors for all recipes
recipe_vectors, vectorizer = generate_recipe_vectors(recipes)

# Sample query
query = ["chicken", 60, "American", "Snack"]

# Retrieve and print top-5 recipes
top_recipes = retrieve_recipes(query, recipe_vectors, vectorizer, recipes, k=5)

for i, recipe in enumerate(top_recipes):
    print(f"Rank {i+1}: {recipe[0]}")
    print(f"Ingredients: {', '.join(recipe[1])}")
    print("Instructions:")
    for step in recipe[2]:
        print(step)
    print(f"Time: {recipe[3]} minutes")
    print(f"Cuisine: {', '.join(recipe[4])}")
    print(f"Course: {recipe[5]}")
    print()

end = time.time()
print(f"Time taken: {(end - start) * 10**3} ms")
