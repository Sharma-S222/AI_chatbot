from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from pyowm import OWM
from datetime import datetime
import os
import psycopg
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

@tool
def metal_price_tool(name: str) -> float :
    """
    This function provides you with the latest price per kg of various metals in USD. 
    You just need to specify the metal name as input, and it will return its current market value. 
    If any unrecognized metal is provided, a ValueError will be raised.
    :param name: metal name
    :return: current per kg price of the metal
    """

    metal_price = {
        # --- Precious Metals ---
        "gold": 132488.1500,
        "rhodium": 257550.0000,
        "iridium": 147890.0000,
        "palladium": 40622.4500,
        "platinum": 54067.8500,
        "ruthenium": 14950.0000,
        "osmium": 12860.0000,
        "silver": 2086.5500,
        "rhenium": 1250.0000,

        # --- Rare Earth & Tech Metals ---
        "terbium": 1150.0000,
        "dysprosium": 380.0000,
        "neodymium": 168.5000,
        "praseodymium": 155.0000,
        "gallium": 285.0000,
        "indium": 265.0000,
        "germanium": 2450.0000,
        "scandium": 3400.0000,
        "hafnium": 1200.0000,
        "yttrium": 45.0000,
        "lanthanum": 8.5000,
        "cerium": 7.2000,
        "samarium": 12.0000,
        "gadolinium": 38.0000,

        # --- Battery & Minor Metals ---
        "cobalt": 56.2900,
        "lithium": 14.1000,
        "molybdenum": 64.5000,
        "magnesium": 3.4500,
        "tungsten": 42.0000,
        "vanadium": 31.5000,
        "bismuth": 24.8000,
        "cadmium": 3.9000,
        "antimony": 22.4000,
        "tellurium": 110.0000,
        "selenium": 33.0000,
        "titanium": 12.5000,
        "zirconium": 35.5000,

        # --- Base & Industrial Metals ---
        "tin": 53.6500,
        "nickel": 17.5300,
        "copper": 14.3300,
        "zinc": 3.5600,
        "aluminum": 3.4000,
        "lead": 1.9500,
        "chromium": 11.2000,
        "manganese": 2.8500,
        "iron": 0.1100
    }
    price = metal_price.get(name, 00)
    return price

@tool
def convert_currency(amount: float, symbol: str, symbol2: str) -> float :
    """Converts amount from one currency to another. 
    3-letter ISO 4217 currency code (e.g., 'EUR', 'INR') should be used.

    Args:
        amount: The monetary value in USD to be converted.
        symbol: The symbol for the current currency of the amount.
        symbol2: The symbol of the current in which it is to be converted.

    Returns:
        The converted amount in the requested currency as a float.
    """
    usd_exchange_value = {
        "USD" : 1,
        "EUR" : 0.8660,  
        "GBP" : 0.7557, 
        "INR" : 94.4600, 
        "JPY" : 161.2800,
        "AUD" : 1.5120, 
        "CAD" : 1.3740,
        "CHF" : 0.8910, 
        "CNY" : 7.2530, 
        "SEK" : 10.5820,
        "NZD" : 1.6980
    }
    usd = usd_exchange_value.get(symbol, 0.0)
    usd_amount = amount/usd
    rate = usd_exchange_value.get(symbol2, 0.0)
    total = usd_amount * rate
    return total

search_tool = ExaSearchRetriever(k = 3)

@tool
def web_search(query: str) -> str :
    """
    Use this tool whenever the user's question requires information that may
    have changed after the model's training data, including:

    - Date and time
    - Current events and news
    - Weather forecasts and conditions
    - Stock prices and market data
    - Live sports results
    - Company updates and announcements
    - Recent technology developments
    - Information about websites or online content

    
    Args:
        query (str): A natural language search query or specific question. Do not include 
                     boolean operators (AND, OR) unless explicitly necessary.
                     
    Returns:
        str: A collection of relevant document snippets, titles, and source URLs.
    """
    result = search_tool.invoke(query)
    return result


api_key= os.getenv("OPENWEATHER_API_KEY")
own = OWM(api_key)
mgr = own.weather_manager()

@tool
def weather_report(city: str) -> dict :
    """Fetches real-time weather details for a specific location from OpenWeather API."""
    observation = mgr.weather_at_place(city)
    weather = observation.weather
    report = weather.to_dict()
    return report


@tool
def current_time():
    """
    call this everytime it is asked for date or time, cause time changes continuously and having the latest time is essential.
    Return current time and date. Call it everytime it is needed. Helps in knowing the latest time with seconds accuracy.
    """
    time = datetime.now().isoformat()
    return time


#For vector searching but as an tool. 
embedding_model = SentenceTransformer("BAAI/bge-m3")
db_url = "postgresql://postgres:ASDFASDF@localhost:5432/postgres"

@tool
def CK_search(query: str) -> dict:
    """
    Look up factual information regarding Google, Apple, and Microsoft
    history, operating systems, hardware strategies, supply chains, and cloud platforms.

    Args: 
        query : the specific search query

    Return:
        dict: A dictionary containing the retrieved text 'context' and a 'confidence_score'
    """
    query_vector = embedding_model.encode(query,normalize_embeddings=True ).tolist()

    matched_paragraphs = []
    best_distance = 2.0

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT p.content, (c.embedding <=> %s::vector) AS dist
            FROM child_chunks c 
            INNER JOIN parent_chunk p ON c.parent_id = p.id
            ORDER BY dist ASC 
            LIMIT 3;
        """, (query_vector,)) 

        rows = cur.fetchall()

    if rows:
        best_distance = rows[0][1]

        for row in rows:
            # row[0] = content, row[1] = distance score
            if row[1] < 0.55:
                matched_paragraphs.append(row[0])

    confidence = max(0.0, 1.0 - best_distance)

    return {
        "context": "\n\n".join(matched_paragraphs) if matched_paragraphs else "No matching data found.",
        "confidence_score": confidence
    }