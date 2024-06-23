import requests
import time 
from bs4 import BeautifulSoup
import csv
import sys
class Crawler:
    pageCount = 76
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    }

    def calculate_total_minutes( text):
        # Split the input text into individual words
        words = text.split()
        
        # Initialize variables to store total hours and minutes
        total_hours = 0
        total_minutes = 0
        
        # Iterate through the words to find numbers followed by 'minutes' and 'hours'
        i = 0
        while i < len(words):
            word = words[i]
            if word.isdigit() and i + 1 < len(words):
                # Check if the next word is 'minutes'
                if words[i + 1].lower() == 'minutes':
                    total_minutes += int(word)
                    i += 1  # Skip the next word ('minutes')
                # Check if the next word is 'hours'
                elif words[i + 1].lower() == 'hour':
                    total_hours += int(word)
                    # If the next word after 'hours' is 'minutes', add those minutes too
                    if i + 2 < len(words) and words[i + 2].lower() == 'minutes':
                        total_minutes += int(words[i + 2])
                        i += 2  # Skip the next two words ('hours' and 'minutes')
            i += 1  # Move to the next word
        
        # Convert hours to minutes and add to the total
        total_minutes += total_hours * 60
        
        return total_minutes


    def fetch_data_from_url(url):
        # Send a GET request to the URL
        response = requests.get(url,headers=Crawler.headers)

        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract data from the HTML content as needed
            # For example, let's extract the ingredients list
            ingredients = soup('div' , class_= 'wprm-recipe-ingredient-group')
            instructions = soup('div' , class_ = 'wprm-recipe-instruction-group')
            cooking_time = soup.find('span' , class_ = 'wprm-recipe-time wprm-block-text-normal')
            Cuisine = soup.find('span', class_='wprm-recipe-cuisine wprm-block-text-normal')
            Course = soup.find('span' , class_ = 'wprm-recipe-course wprm-block-text-normal')
            if ingredients and cooking_time and instructions and Cuisine and Course: 
                
                ingredients_list = [ ingredient.text for ingredient in ingredients]
                instructions_list = [instruction.text for instruction in instructions]
                return ingredients_list ,Crawler.calculate_total_minutes(cooking_time.text) , instructions_list ,Cuisine.text , Course.text
                
        else:
            print(f"Failed to fetch data from URL: {url}")
            return None



    def extract_links_info(elements):
        recipes = []
        for element in elements:
            link = element.find('a')
            if link:
                href = link.get('href')
                name = link.text.strip()
                # Check if the name contains any numbers
                if not any(char.isdigit() for char in name):
                    data = Crawler.fetch_data_from_url(href)
                    if data:
                        ingredients, time , instructions , Cuisine, Course = data  # Unpack the data if it's not None
                        if ingredients and time and instructions and Cuisine:
                            recipes.append((name, ingredients, time ,instructions, Cuisine,Course))
                    else:
                        # Handle case where fetch_data_from_url returned None
                        print(f"Failed to fetch data for recipe: {name}")
        return recipes


    def crawl():    
        for page in range(2,Crawler.pageCount):
            url = "https://www.recipetineats.com/recipes/?fwp_paged=" + str(page)
            print(page)
            response = requests.get(url=url,headers=Crawler.headers)
            # Results = []
            soup = BeautifulSoup(response.text , 'html.parser')
            resultsRecipes = soup.find_all('h2' , class_ = 'entry-title')
            resutsRecipesList = list(resultsRecipes)
            recipes = Crawler.extract_links_info(resutsRecipesList)
            if recipes :
                with open('recipes2.csv', 'a', newline='\n', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerows(recipes)


if __name__ == '__main__':  
    Crawler.crawl()
