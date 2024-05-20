from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
import math
import json
import re


class UserMatcher:
    def __init__(self, model, user_data):
        """
        Initialize the UserMatcher class.

        Args:
        - model: Word embeddings model for computing word similarities.
        - user_data (dict): Data of the user to be matched.
        """

        # MongoDB connection URL
        self.MONGO_URL = "mongodb://localhost:27017"

        # Connect to MongoDB
        self.client = AsyncIOMotorClient(self.MONGO_URL)

        # Select database
        self.db = self.client["Users"]

        # Select collection
        self.collection = self.db["Details"]

        # Load word embeddings model
        self.model = model

        # User data
        self.user_data = user_data

        self.similar_topics = {}

        self.common_attributes = []

        self.penalty = 0

        self.self_similarity = 0

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two points on the Earth's surface using the Haversine formula.

        Parameters:
            lat1 (float): Latitude of the first point in degrees.
            lon1 (float): Longitude of the first point in degrees.
            lat2 (float): Latitude of the second point in degrees.
            lon2 (float): Longitude of the second point in degrees.

        Returns:
            distance (float): The distance between the two points in kilometers.
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = 6371 * c  # Earth radius in kilometers

        return distance

    def get_values(self, user, keys):
        """
        Extract values from a dictionary based on provided keys.

        Args:
        - user (dict): The dictionary to extract values from.
        - keys (list): List of keys to extract values for.

        Returns:
        - array (list): List containing values extracted from the dictionary.
        """
        array = []
        for key in keys:
            if isinstance(user[key], str):
                array.append(user[key])
            elif isinstance(user[key], list):
                array.extend(user[key])
            elif isinstance(user[key], dict):
                array.extend(self.get_values(user[key], user[key].keys()))
        return array

    def get_attributes(self, user):
        """
        Extracts all attributes from a user dictionary, handling any depth.

        Args:
        - user (dict): The user dictionary.

        Returns:
        - attributes (list): List containing all attributes from the user dictionary.
        """
        attributes = []
        def extract_values(data):
            for key, value in data.items():
                if key != "name":
                    if isinstance(value, list):
                        attributes.extend(value)
                    elif isinstance(value, dict):
                        extract_values(value)
                    else:
                        attributes.append(value)

        extract_values(user)
        return attributes

    def word_similarity(self, word1, word2):
        """
        Compute the similarity between two words using cosine similarity of their word embeddings.

        Args:
        - word1 (str): First word.
        - word2 (str): Second word.

        Returns:
        - similarity (float): Cosine similarity between the word embeddings of the two words.
        """
        try:
            # Get word embeddings
            vec1 = self.model[word1]
            vec2 = self.model[word2]
            # Compute cosine similarity
            similarity = cosine_similarity([vec1], [vec2])[0][0]

            similar = self.similar_topics.keys()

            word_compare = f"{word2} {word1}"

            if word_compare not in similar or word1 == word2:
                self.similar_topics[f"{word1} {word2}"] = similarity
            return similarity
        except KeyError:
            # Handle out-of-vocabulary words
            return 0.0

    def array_similarity(self, array1, array2):
        """
        Compute the similarity between two arrays of words using cosine similarity of their word embeddings.

        Args:
        - array1 (list): First array of words.
        - array2 (list): Second array of words.

        Returns:
        - overall_similarity (float): Average cosine similarity between all pairs of words from the two arrays.
        """
        overall_similarity = 0.0
        # Compute pairwise similarities between words in the arrays
        if len(array1) > 0 & len(array2) > 0:
            similarities = np.zeros((len(array1), len(array2)))
            for i, word1 in enumerate(array1):
                for j, word2 in enumerate(array2):
                    #if i >= j:
                    similarities[i][j] = self.word_similarity(word1, word2)
       
            array = similarities.flatten()


            overall_similarity = np.mean(array)

        if math.isnan(overall_similarity):
            return 0
        else:
            return overall_similarity

    def dict_similarity(self, dict1, dict2):
        """
        Calculate the similarity between two dictionaries.

        Args:
        - dict1 (dict): First input dictionary.
        - dict2 (dict): Second input dictionary.

        Returns:
        - overall_similarity (float): Overall similarity between the dictionaries.
        """
        # Initialize overall similarity
        overall_similarity = 0.0

        try:
            # Compute pairwise similarities between keys in the dictionaries
            similarities = np.zeros((len(dict1), len(dict2)))
            for i, (key1, value1) in enumerate(dict1.items()):
                for j, (key2, value2) in enumerate(dict2.items()):
                    # If both values are dictionaries, recursively compute their similarity
                    if isinstance(value1, dict) and isinstance(value2, dict):
                        similarities[i][j] = self.dict_similarity(value1, value2)
                    # Otherwise, compute the similarity using a custom function (word_similarity)
                    elif isinstance(value1, list) and isinstance(value2, list):
                        similarities[i][j] = self.array_similarity(value1, value2)
                    else:
                        similarities[i][j] = self.word_similarity(value1, value2)
            # Aggregate the similarities
            overall_similarity = np.mean(similarities)
            if math.isnan(overall_similarity):
                return 0
            else:
                return overall_similarity
        except Exception as e:
            return 0

    def set_distance(self, user, users):
        """
        Calculate and set the distance between a user and a list of users.

        Args:
        - user (dict): User whose distance is to be calculated.
        - users (list): List of users to calculate distances to.

        Returns:
        - nearby (list): List of users with distances set.
        """
        nearby = []
        for individual in users:
            individual["distance"] = round(self.calculate_distance(user["location"][0],user["location"][1],individual["location"][0],individual["location"][1]),2)
            nearby.append(individual)
        return nearby

    def normalize_score(self, score_dict, num_keys):
        """
        Normalize a score between 0 and 1 based on the number of keys.

        Args:
        - score (float): The score to be normalized.
        - num_keys (int): The total number of keys.

        Returns:
        - normalized_score (float): The normalized score between 0 and 1.
        """
        # Check if num_keys is greater than 0 to avoid division by zero
        score = 0
        num = num_keys
        for key, value in score_dict.items():
            if value == 0:
                num = num - 1
            score = score + value
        normalized_score = score / num

        return round(normalized_score,2)

    # For non common elements
    def exponential_decay(self, x):
        """
        Calculate the exponential decay function for a given value.

        Args:
        - x (float): Input value.

        Returns:
        - result (float): Result of the exponential decay function.
        """
        return np.exp(-x + 1) + np.log2(x+1)
    
    # For commom elements of a field
    def exp_log(self, x):
        """
        Calculate the exponential of the logarithm of a given value.

        Args:
        - x (float): Input value.

        Returns:
        - result (float): Result of the exponential of the logarithm.
        """
        return math.exp(math.log(x+1))
    
    def contains_digits(self,item):
        return bool(re.search(r'\d', item))
    
    def contains_link(self, item):
        # Regular expression to match URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return bool(re.search(url_pattern, item))
    
    def similarity_score(self, user1, user2):
        similarity_dict = {}
        common_keys = set(user1) & set(user2)
        for key in common_keys:
            if key == "location":
                pass
            elif key == "_id":
                pass
            else:
                if isinstance(user1[key], str) and isinstance(user2[key], str):
                    similarity_dict[key] = self.word_similarity(user1[key], user2[key])
                elif isinstance(user1[key], list) and isinstance(user2[key], list):
                    similarity_dict[key] = self.array_similarity(user1[key], user2[key])
                elif isinstance(user1[key], dict) and isinstance(user2[key], dict):
                    similarity_dict[key] = self.dict_similarity(user1[key], user2[key])

        user1_attributes = self.get_attributes(user1)
        user2_attributes = self.get_attributes(user2)

        # Elements in a but not in b (a-b)
        user_only = [x for x in user1.keys() if x != 'distance' and x not in user2.keys()]

        # Elements in b but not in a (b-a)
        top_user_only = [x for x in user2.keys() if x != 'distance' and x not in user1.keys()]

        self.penalty = max(len(user_only), len(top_user_only))

        unmatched_weights = self.exponential_decay(max(len(user_only), len(top_user_only)))
        unmatched_similarity = self.array_similarity(user_only, top_user_only)

        intersection = [item for item in set(user1_attributes) & set(user2_attributes) if not isinstance(item, (int, float)) and not self.contains_digits(str(item)) and not self.contains_link(str(item))][:5]

        self.common_attributes = intersection
        matched_weights = self.exp_log(len(intersection))
        matched_similarity = self.normalize_score(similarity_dict, len(common_keys))
        
        total_similarity = (matched_similarity*matched_weights) + (unmatched_similarity * unmatched_weights)

        return round(total_similarity,2)

    def user_proximity(self, users, proximity):
        return [item for item in users if item.get("distance") <= proximity]

    def matched_user(self, user, users, proximity, max_proximity, threshold, min_threshold):
        while True:
            near_user = self.user_proximity(users, proximity)
            
            for individual in near_user:
                individual["similarity"] = self.similarity_score(self.user_data, individual)
            
            if len(near_user) > 0:
                return near_user
            
            if proximity < max_proximity:
                proximity += 1

            elif threshold > min_threshold:
                threshold -= 0.1

            else:
                return []
        
    async def get_users(self):
        proximity = 2
        max_proximity = 10
        threshold = 0.5
        min_threshold = 0

        self.self_similarity = self.similarity_score(self.user_data, self.user_data)

        users = await self.collection.find().to_list(length=None)

        nearby_users = self.set_distance(self.user_data, users)

        users = self.matched_user(user=self.user_data, users=nearby_users, proximity=proximity, max_proximity=max_proximity, threshold=threshold, min_threshold=min_threshold)

        if len(users):
            top_user = max(users, key=lambda x: x['similarity'])
            # Get the top 5 values from the dictionary
            top_values = sorted(self.similar_topics.values(), reverse=True)[:5]
            
            # Filter the dictionary to keep only the key-value pairs corresponding to the top 5 values
            filtered_dict = {key: value for key, value in self.similar_topics.items() if value in top_values}
                
            first_elements = [key.split()[0] for key in filtered_dict.keys()]

            filtered_first_elements = [element for element in first_elements if not (element.isdigit())]

            similarity = min(round((top_user["similarity"] / self.self_similarity), 2), 1.00)
            data = {}
            data["userId"] = top_user["_id"]
            data["similarity"] =  similarity - (min(self.penalty, 30) / 100) if similarity == 1.0 else similarity 
            data["distance"] = top_user["distance"]
            data["common_attributes"] = list(set([element for element in self.common_attributes if not (element.isdigit())] + filtered_first_elements))
            
            users = [json.loads(json.dumps(data, default=str))]
            return users[0]
        else:        
            return users