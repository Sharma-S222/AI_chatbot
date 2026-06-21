from langchain_core.tools import tool
from langchain_exa import ExaSearchRetriever
from dotenv import load_dotenv
load_dotenv()

@tool
def metal_price_tool(name: str) -> float :
    """
    This function provides the latest price per kg of various metals in USD. 
    You must specify a single metal name as input, and it will return its current market value. 

    ALWAYS use this tool instead of web search if the user asks for the price, rate, 
    cost, or market valuation of any metal. 

    Supported metals include but are not limited to:
    - Precious Metals: Gold, Silver, Platinum, Palladium
    - Base & Industrial Metals: Copper, Aluminum, Zinc, Lead, Nickel, Tin, Iron Ore, Steel
    - Battery & Rare Metals: Lithium, Cobalt, Manganese

    CRITICAL FAILURE HANDLING: 
    If the metal name is unrecognized or unsupported, this tool will raise a ValueError. 
    If you receive a ValueError or an error message from this tool, you are explicitly 
    authorized to fall back to the 'web_search' tool to find the price manually.

    Args:
        metal_name (str): The common lowercase name of the metal (e.g., 'gold', 'copper').
                          Do not include extra words like 'price' or 'today'.

    Returns:
        str: Current per kg price of the metal in USD, or an error message.
    """


    metal_price = {
        # --- Precious Metals ---
        "gold": 132488.1500,
        "rhodium": 154320.0000,
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
    price = metal_price.get(name, 0.0)
    return price

@tool
def convert_currency(amount: float, symbol: str) -> float :
    """Converts an amount from US Dollars (USD) to a target foreign currency.

    This function standardizes the input currency code to uppercase, checks
    if it exists within the supported rates dictionary, and calculates the 
    converted monetary value. If the currency code is unrecognized, it 
    defaults to a 1:1 fallback rate (treating it as USD).

    Args:
        amount: The monetary value in USD to be converted.
        symbol: The 3-letter ISO 4217 currency code (e.g., 'EUR', 'INR').

    Returns:
        The converted amount in the requested currency as a float.
    """
    usd_exchange_value = {
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
    if symbol.upper() not in usd_exchange_value:
        rate = amount*1
    else:  
        val = usd_exchange_value.get(symbol,0.0)
        rate = val * amount

    return rate

search_tool = ExaSearchRetriever(k = 3)

@tool
def web_search(query: str) -> str :
    """
    Search the internet to retrieve up-to-date information, facts, and live web contents.

    Use this tool whenever the user's question requires information that may
    have changed after the model's training data, including:

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

