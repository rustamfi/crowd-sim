"""
SF Crowd Voting Simulator — Population Generator
=================================================
Fetches ACS 2022 5-year PUMS microdata for San Francisco PUMAs 07501-07507,
weighted-samples 30 adult residents, enriches each with neighborhood, occupation,
name, OCEAN personality scores, and a behavioral profile, then writes to
results/agents.json.

REQ-001 through REQ-013.

OCEAN Demographic Derivation (REQ-008)
--------------------------------------
Scores are DERIVED from demographic correlates, not assigned randomly. Each
trait = base 5.5 + Σ(demographic terms) + small individual noise. Demographics
are the PRIMARY, dominant signal; noise (gauss 0, 0.7) adds within-group variety
so two similar residents are not identical. Directions and relative magnitudes
follow the personality-psychology literature (age "maturity principle", sex
differences in N/A, SES, occupational selection).

Age is CONTINUOUS via age_factor = (age - 45) / 17 (~ -1.4 young to +2.0 old):
  O -0.5  C +0.7  E -0.3  A +0.6  N -0.7  per unit  (older = more C/A, less N/O/E)
Household income is CONTINUOUS via ses = (log10(household_income) - 5.08) / 0.5,
floored at $15k. Pivot 5.08 ~ $120k ~ SF median household income, so ses=0 is a
typical SF household:
  O +0.3  C +0.5  E +0.3  A -0.1  N -0.5  per unit  (higher SES = more C, less N)
Sex = Female: N +0.9  A +0.6  C +0.2  O +0.1   Male: N -0.4  A -0.2
Tenure = Own:  C +0.4  A +0.2  N -0.4          Rent: O +0.2  C -0.2  E +0.2  N +0.3
Occupation category (broad keyword match; every worker gets a signal):
  tech/engineering  O +0.8  C +0.4  E -0.2  A -0.2
  healthcare        C +0.6  E +0.2  A +0.8
  arts/media        O +1.2  C -0.3  E +0.3  N +0.3
  education         O +0.4  C +0.5  E +0.2  A +0.6
  business/mgmt     O +0.2  C +0.6  E +0.5  A -0.2  N -0.2
  service/food      C -0.1  E +0.4  A +0.5  N +0.2
  office/admin      C +0.4  E -0.2
  (not in labor force / unmatched: no occupation term)
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_PATH = DATA_DIR / "pums_sf.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "agents.json"

# ---------------------------------------------------------------------------
# Agent-memory (delivery-app experiences) — optional, LLM-backed
# ---------------------------------------------------------------------------
# Population generation is LLM-free by default. The ONLY path that calls the
# LLM is the optional agent-memory feature (use_memory=True), which asks
# o4-mini for 3 realistic past food-delivery-app experiences per agent. The
# experiences become part of the persona at vote time (run_scenario.py) and so
# influence how the agent votes. Concurrency mirrors run_scenario.py.
MEMORY_MODEL = "o4-mini"
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "10"))
DELIVERY_EXPERIENCE_COUNT = 3

# ---------------------------------------------------------------------------
# Census API — REQ-001
# ---------------------------------------------------------------------------
CENSUS_URL = "https://api.census.gov/data/2022/acs/acs5/pums"
_census_key = os.environ.get("CENSUS_API_KEY", "")
# PUMS API uses PUMA20 as a variable (not a geography filter).
# We fetch all CA records and filter to SF PUMAs in Python.
# 2020 PUMA codes for San Francisco County (FIPS 06075)
SF_PUMAS = {"07507", "07508", "07509", "07510", "07511", "07512", "07513", "07514"}
CENSUS_PARAMS = {
    "get": "AGEP,OCCP,HINCP,TEN,RAC1P,HISP,SEX,PWGTP,PUMA20",
    "for": "state:06",
    **({"key": _census_key} if _census_key else {}),
}

# ---------------------------------------------------------------------------
# PUMA-to-Neighborhood mapping — REQ-005
# Each PUMA maps to a list of plausible SF neighborhood names.
# On agent creation, one neighborhood is chosen deterministically via per-agent RNG.
# ---------------------------------------------------------------------------
PUMA_NEIGHBORHOODS = {
    # 2020 PUMA definitions for San Francisco County (FIPS 06075)
    "07507": ["Mission", "Potrero Hill", "Bernal Heights"],
    # 07507: Mission District and adjacent hilltop neighborhoods
    "07508": ["Bayview", "Visitacion Valley", "Excelsior", "Portola"],
    # 07508: southeastern SF, diverse working-class neighborhoods
    "07509": ["SoMa", "Financial District", "Tenderloin", "South Beach"],
    # 07509: downtown core, dense urban mix of tech offices and SROs
    "07510": ["Chinatown", "Nob Hill", "North Beach", "Russian Hill"],
    # 07510: northeast hill neighborhoods, historic and tourist corridor
    "07511": ["Richmond", "Sunset", "Parkside"],
    # 07511: western residential neighborhoods, foggy avenues
    "07512": ["Marina", "Pacific Heights", "Presidio Heights", "Cow Hollow"],
    # 07512: affluent northern neighborhoods along the bay
    "07513": ["Haight-Ashbury", "Castro", "Noe Valley", "Cole Valley"],
    # 07513: central-south neighborhoods, LGBTQ+ and bohemian culture
    "07514": ["Twin Peaks", "Glen Park", "Diamond Heights", "West Portal"],
    # 07514: hilltop and valley neighborhoods in central-south SF
}

# ---------------------------------------------------------------------------
# OCCP Code-to-Occupation mapping — REQ-006
# Covers the top ~40+ most frequent codes in SF PUMS plus special values.
# Unknown codes fall back to range-based categories defined in OCCP_RANGES.
# ---------------------------------------------------------------------------
OCCP_LOOKUP = {
    # Management
    "10":   "Chief Executive",
    "20":   "General/Operations Manager",
    "51":   "Marketing Manager",
    "52":   "Sales Manager",
    "60":   "Operations Manager",
    "100":  "Administrative Manager",
    "110":  "Facilities Manager",
    "120":  "HR Manager",
    "136":  "Financial Manager",
    "140":  "IT Manager",
    # Business/Finance
    "500":  "Business Analyst",
    "510":  "Buyer/Purchasing Agent",
    "520":  "HR Specialist",
    "530":  "Training Specialist",
    "540":  "Logistician",
    "565":  "Project Manager",
    "600":  "Accountant",
    "620":  "Financial Analyst",
    "630":  "Budget Analyst",
    "700":  "Loan Officer",
    "710":  "Tax Examiner",
    "800":  "Market Research Analyst",
    "840":  "Management Analyst",
    # Computer/Math — Software engineer codes capped at 4 (REQ-011)
    "1005": "Computer Systems Analyst",
    "1006": "Computer Systems Analyst",
    "1007": "Data Scientist",
    "1010": "Software Developer",        # REQ-011 cap codes
    "1020": "Software Developer",        # REQ-011 cap codes
    "1030": "Software Developer",        # REQ-011 cap codes
    "1050": "Software Engineer",         # REQ-011 cap codes
    "1060": "Software Engineer",         # REQ-011 cap codes
    "1100": "Network Administrator",
    "1105": "Network Administrator",
    "1110": "Network/Computer Systems Admin",
    "1200": "Database Administrator",
    "1220": "Information Security Analyst",
    "1240": "Data Scientist",
    # Engineering
    "1300": "Civil Engineer",
    "1310": "Architect",
    "1320": "Aerospace Engineer",
    "1340": "Biomedical Engineer",
    "1360": "Electrical Engineer",
    "1400": "Mechanical Engineer",
    "1420": "Chemical Engineer",
    "1440": "Environmental Engineer",
    "1460": "Industrial Engineer",
    "1530": "Drafter",
    "1550": "Engineering Technician",
    # Science
    "1600": "Agricultural Scientist",
    "1610": "Biological Scientist",
    "1640": "Conservation Scientist",
    "1650": "Medical Scientist",
    "1700": "Economist",
    "1710": "Survey Researcher",
    "1720": "Psychologist",
    "1740": "Urban Planner",
    "1760": "Sociologist",
    "1800": "Chemist",
    "1820": "Physicist",
    "1860": "Physical Scientist",
    "1900": "Agricultural Technician",
    "1930": "Lab Technician",
    "1960": "Social Science Research Assistant",
    "1980": "Research Assistant",
    # Social Service
    "2000": "Social Worker",
    "2010": "Substance Abuse Counselor",
    "2015": "Mental Health Counselor",
    "2020": "Community Health Worker",
    "2040": "Social and Human Service Assistant",
    "2050": "Probation Officer",
    "2060": "Religious Worker",
    # Legal
    "2100": "Lawyer",
    "2105": "Judicial Law Clerk",
    "2110": "Paralegal",
    "2145": "Title Examiner",
    "2160": "Court Reporter",
    "2170": "Legal Support",
    "2180": "Legal Secretary",
    # Education
    "2200": "Postsecondary Teacher",
    "2210": "Preschool Teacher",
    "2300": "Elementary School Teacher",
    "2310": "Middle School Teacher",
    "2320": "High School Teacher",
    "2330": "Special Education Teacher",
    "2340": "Vocational Ed Teacher",
    "2360": "Substitute Teacher",
    "2400": "Librarian",
    "2545": "Teaching Assistant",
    "2550": "Instructional Coordinator",
    # Arts/Media
    "2600": "Artist",
    "2630": "Fashion Designer",
    "2640": "Graphic Designer",
    "2650": "Interior Designer",
    "2700": "Actor",
    "2710": "Producer/Director",
    "2720": "Athlete",
    "2740": "Dancer",
    "2750": "Musician",
    "2760": "Photographer",
    "2800": "News Reporter",
    "2810": "Editor",
    "2820": "Technical Writer",
    "2830": "Public Relations Specialist",
    "2840": "Advertising Manager",
    "2850": "Marketing Specialist",
    "2860": "Interpreter",
    "2900": "Broadcast Technician",
    "2910": "Sound Engineer",
    "2960": "Web Developer",
    # Healthcare
    "3000": "Physician",
    "3010": "Surgeon",
    "3030": "Psychiatrist",
    "3050": "Family Doctor",
    "3090": "Pharmacist",
    "3100": "Optometrist",
    "3110": "Physical Therapist",
    "3120": "Occupational Therapist",
    "3140": "Speech Therapist",
    "3150": "Audiologist",
    "3160": "Radiologist",
    "3200": "Registered Nurse",
    "3210": "Nurse Practitioner",
    "3220": "Licensed Practical Nurse",
    "3230": "Nurse Midwife",
    "3245": "Nurse Anesthetist",
    "3250": "Nutritionist",
    "3255": "Therapist",
    "3300": "Clinical Lab Technician",
    "3322": "Dental Hygienist",
    "3323": "Dentist",
    "3400": "Paramedic",
    "3500": "Veterinarian",
    "3550": "Veterinary Technician",
    # Healthcare Support
    "3600": "Dental Assistant",
    "3610": "Medical Assistant",
    "3620": "Home Health Aide",
    "3630": "Personal Care Aide",
    "3640": "Medical Transcriptionist",
    "3645": "Phlebotomist",
    "3655": "Medical Secretary",
    # Protective Service
    "3700": "Police Officer",
    "3710": "Firefighter",
    "3720": "Fire Inspector",
    "3730": "Bailiff",
    "3740": "Correctional Officer",
    "3801": "Security Guard",
    "3960": "Crossing Guard",
    # Food Service
    "4000": "Chef",
    "4010": "Baker",
    "4020": "Food Preparation Worker",
    "4030": "Bartender",
    "4040": "Combined Food Prep & Server",
    "4050": "Counter Attendant",
    "4060": "Waiter/Waitress",
    "4110": "Dishwasher",
    "4120": "Restaurant Host",
    "4160": "Food Server",
    # Cleaning/Maintenance
    "4200": "Janitor",
    "4210": "Maid/Housekeeping Cleaner",
    "4220": "Pest Control Worker",
    "4230": "Grounds Maintenance Worker",
    "4255": "Building Cleaner",
    # Personal Care
    "4300": "Childcare Worker",
    "4320": "Hairdresser",
    "4340": "Recreation Worker",
    "4350": "Fitness Trainer",
    "4400": "Funeral Attendant",
    "4420": "Tour Guide",
    "4460": "Animal Trainer",
    "4655": "Personal Care Worker",
    # Sales
    "4700": "Retail Salesperson",
    "4710": "Sales Representative",
    "4720": "Real Estate Agent",
    "4740": "Insurance Sales Agent",
    "4760": "Financial Services Sales Agent",
    "4800": "Cashier",
    "4820": "Counter Sales Clerk",
    "4840": "Rental Clerk",
    "4850": "Telemarketer",
    "4965": "Sales Support Worker",
    # Office/Admin
    "5000": "Office Clerk",
    "5010": "Bookkeeping Clerk",
    "5020": "Payroll Clerk",
    "5030": "Billing Clerk",
    "5100": "Secretary",
    "5110": "Data Entry Keyer",
    "5120": "Word Processor",
    "5200": "Mail Clerk",
    "5220": "Shipping/Receiving Clerk",
    "5230": "Weigher/Measurer",
    "5240": "Meter Reader",
    "5300": "Customer Service Representative",
    "5310": "Phone Operator",
    "5400": "File Clerk",
    "5410": "Records Clerk",
    "5420": "Human Resources Clerk",
    "5500": "Office Machine Operator",
    "5700": "Production Planner",
    "5800": "Executive Assistant",
    "5900": "Administrative Assistant",
    "5940": "Office Supervisor",
    # Farming
    "6005": "Farmer",
    "6010": "Agricultural Laborer",
    "6050": "Agricultural Inspector",
    "6130": "Forest/Conservation Worker",
    # Construction
    "6200": "Construction Manager",
    "6210": "Construction Worker",
    "6220": "Brickmason",
    "6230": "Carpenter",
    "6240": "Cement Mason",
    "6260": "Construction Laborer",
    "6300": "Electrician",
    "6330": "Glazier",
    "6355": "Insulation Worker",
    "6360": "Painter",
    "6400": "Plumber",
    "6440": "Roofer",
    "6500": "Iron Worker",
    "6515": "Solar Panel Installer",
    "6600": "Construction Inspector",
    "6700": "Highway Maintenance Worker",
    "6800": "Surveyor",
    "6950": "Construction Equipment Operator",
    # Repair/Maintenance
    "7000": "Mechanic",
    "7010": "Aircraft Mechanic",
    "7020": "Automotive Body Repairer",
    "7030": "Auto Service Technician",
    "7040": "Bus/Truck Mechanic",
    "7100": "Electrical Equipment Installer",
    "7200": "HVAC Mechanic",
    "7260": "Industrial Machinery Mechanic",
    "7300": "Maintenance Worker",
    "7315": "Millwright",
    "7400": "Computer Repair Technician",
    "7410": "Office Machine Repairer",
    "7620": "Tailor",
    "7640": "Upholsterer",
    # Production
    "7700": "Machinist",
    "7710": "Tool and Die Maker",
    "7800": "Welder",
    "7900": "Laundry Worker",
    "8000": "Food Processing Worker",
    "8200": "Printing Worker",
    "8220": "Bindery Worker",
    "8300": "Assembler",
    "8530": "Packaging Machine Operator",
    "8710": "Medical Equipment Preparer",
    "8740": "Quality Inspector",
    "8990": "Production Worker",
    # Transportation
    "9000": "Pilot",
    "9030": "Air Traffic Controller",
    "9110": "Bus Driver",
    "9120": "Taxi/Uber Driver",
    "9130": "Driver/Sales Worker",
    "9140": "Truck Driver",
    "9200": "Locomotive Engineer",
    "9260": "Railroad Conductor",
    "9300": "Sailor",
    "9350": "Ship Engineer",
    "9410": "Material Moving Worker",
    "9420": "Crane Operator",
    "9500": "Parking Lot Attendant",
    "9510": "Service Station Attendant",
    "9520": "Traffic Technician",
    "9560": "Crossing Guard",
    "9600": "Cleaner of Vehicles",
    "9610": "Refuse Collector",
    "9620": "Laborer",
    "9640": "Stock/Order Filler",
    "9645": "Pumping Station Operator",
    "9720": "Postal Worker",
    "9760": "Courier/Messenger",
    # Special values
    "N":    "Not in labor force",
    "9920": "Unemployed",
}

# Codes that count as software engineers for REQ-011 cap
SWE_CODES = {"1010", "1020", "1030", "1050", "1060"}

# OCCP range-based fallback — architecture.md OCCP_RANGES
OCCP_RANGES = [
    (10,   440,   "Management"),
    (500,  960,   "Business/Finance"),
    (1005, 1240,  "Computer/Math"),
    (1300, 1560,  "Engineering"),
    (1600, 1980,  "Science"),
    (2000, 2060,  "Social Service"),
    (2100, 2180,  "Legal"),
    (2200, 2550,  "Education"),
    (2600, 2960,  "Arts/Media"),
    (3000, 3550,  "Healthcare"),
    (3600, 3655,  "Healthcare Support"),
    (3700, 3960,  "Protective Service"),
    (4000, 4160,  "Food Service"),
    (4200, 4255,  "Cleaning/Maintenance"),
    (4300, 4655,  "Personal Care"),
    (4700, 4965,  "Sales"),
    (5000, 5940,  "Office/Admin"),
    (6005, 6130,  "Farming"),
    (6200, 6950,  "Construction"),
    (7000, 7640,  "Repair/Maintenance"),
    (7700, 8990,  "Production"),
    (9000, 9760,  "Transportation"),
]

# ---------------------------------------------------------------------------
# Ethnicity-stratified name lists — REQ-007
# Keyed by (rac1p_group, sex): "male" or "female".
# HISP > 1 overrides race and uses the "hispanic" group.
# rac1p_group: "white", "black", "asian", "hispanic", "aian", "nhpi", "other"
# ---------------------------------------------------------------------------
FIRST_NAMES = {
    ("white", "male"): [
        "James", "John", "Robert", "Michael", "William", "David", "Richard",
        "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
        "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
        "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward",
        "Jason", "Jeffrey", "Ryan",
    ],
    ("white", "female"): [
        "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Susan", "Jessica",
        "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
        "Ashley", "Dorothy", "Kimberly", "Emily", "Donna", "Michelle",
        "Carol", "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca",
        "Sharon", "Laura", "Cynthia", "Kathleen",
    ],
    ("black", "male"): [
        "DeShawn", "Malik", "Darius", "Jalen", "Terrence", "Marcus",
        "DeAndre", "Jabari", "Kwame", "Lamar", "Devonte", "Andre",
        "Isaiah", "Elijah", "Tyrone", "Jerome", "Calvin", "Derrick",
        "Darnell", "Antoine", "Cedric", "Reginald", "Kendrick", "LeRoy",
        "Marlon", "Damien", "Desmond", "Trayvon", "Marquise", "Bryson",
    ],
    ("black", "female"): [
        "Aaliyah", "Imani", "Keisha", "Latoya", "Shanice", "Tanisha",
        "Ebony", "Jasmine", "Monique", "Destiny", "Brianna", "Naomi",
        "Zora", "Denise", "Nichelle", "Tiffany", "Raven", "Diamond",
        "Chantel", "Tamara", "Lakisha", "Shonda", "Unique", "Deja",
        "Iesha", "Latasha", "Sharonda", "Shaniqua", "Jalisa", "Precious",
    ],
    ("asian", "male"): [
        "Wei", "Ming", "Jae", "Kevin", "Jason", "Daniel", "Brian",
        "Tony", "Eric", "Ryan", "Andy", "Steven", "Michael", "Allen",
        "Victor", "Brandon", "Timothy", "Nathan", "Derek", "Jeffrey",
        "Winston", "Raymond", "Dennis", "Eugene", "Harold", "Stanley",
        "Norman", "Walter", "Chester", "Hiro",
    ],
    ("asian", "female"): [
        "Li", "Mei", "Yuki", "Priya", "Aisha", "Jenny", "Amy",
        "Christine", "Grace", "Helen", "Irene", "Joyce", "Karen",
        "Linda", "Nancy", "Patricia", "Ruth", "Sandra", "Sharon",
        "Tina", "Susan", "Connie", "Diane", "Elaine", "Fiona",
        "Gloria", "Hannah", "Ivy", "Janet", "Katherine",
    ],
    ("hispanic", "male"): [
        "Jose", "Juan", "Carlos", "Luis", "Miguel", "Angel", "Francisco",
        "Jorge", "Antonio", "Manuel", "Ricardo", "Eduardo", "Roberto",
        "Alejandro", "Sergio", "Fernando", "Diego", "Gabriel", "Rafael",
        "Hector", "Mario", "Raul", "Arturo", "Javier", "Ernesto",
        "Marco", "Cesar", "Victor", "Adrian", "Ruben",
    ],
    ("hispanic", "female"): [
        "Maria", "Ana", "Carmen", "Rosa", "Elena", "Gloria", "Patricia",
        "Martha", "Sandra", "Isabel", "Diana", "Claudia", "Veronica",
        "Adriana", "Lucia", "Graciela", "Esperanza", "Yolanda", "Dolores",
        "Silvia", "Alicia", "Sofia", "Valentina", "Gabriela", "Camila",
        "Fernanda", "Daniela", "Mariana", "Natalia", "Lorena",
    ],
    ("aian", "male"): [
        "Dakota", "Chase", "Hunter", "Travis", "Nathan", "Tyler", "Cody",
        "Beau", "Cole", "Dylan", "Wyatt", "Levi", "Elias", "Aaron",
        "Isaac", "Caleb", "Jesse", "Seth", "Luke", "Owen",
    ],
    ("aian", "female"): [
        "Skylar", "Dakota", "Sierra", "Savannah", "Cheyenne", "Hailey",
        "Cassidy", "Amber", "Dawn", "Autumn", "Sage", "Willow", "Raven",
        "Crystal", "Jade", "Ruby", "Pearl", "Luna", "Nova", "Aurora",
    ],
    ("nhpi", "male"): [
        "Tane", "Keanu", "Kaimana", "Lono", "Alika", "Makoa", "Noa",
        "Tavita", "Fetu", "Sione", "Tevita", "Peni", "Sela", "Ioane",
        "Samisoni", "Tui", "Vili", "Malo", "Fili", "Pio",
    ],
    ("nhpi", "female"): [
        "Leilani", "Kaimana", "Haunani", "Moana", "Maile", "Puanani",
        "Alohilani", "Noelani", "Kalani", "Hina", "Sina", "Faleolo",
        "Lotu", "Malia", "Sela", "Losa", "Fia", "Nita", "Lina", "Tina",
    ],
    ("other", "male"): [
        "Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Jamie",
        "Cameron", "Logan", "Avery", "Parker", "Quinn", "Reese", "Drew",
        "Finley", "Hayden", "Hunter", "Skyler", "Spencer", "Peyton",
    ],
    ("other", "female"): [
        "Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Jamie",
        "Cameron", "Avery", "Parker", "Quinn", "Reese", "Drew", "Finley",
        "Hayden", "Skyler", "Spencer", "Peyton", "Robin", "Sage",
    ],
}

LAST_NAMES = {
    "white": [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
        "Wilson", "Anderson", "Taylor", "Thomas", "Jackson", "White", "Harris",
        "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark",
    ],
    "black": [
        "Jackson", "Washington", "Harris", "Robinson", "Walker", "Johnson",
        "Williams", "Brown", "Jones", "Davis", "Thomas", "Thompson", "Moore",
        "White", "Scott", "Lewis", "Adams", "Green", "Hall", "Nelson",
    ],
    "asian": [
        "Chen", "Wang", "Kim", "Li", "Zhang", "Liu", "Park", "Nguyen",
        "Lee", "Yang", "Wu", "Lin", "Huang", "Cheng", "Lam", "Chan",
        "Patel", "Singh", "Kumar", "Suzuki",
    ],
    "hispanic": [
        "Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Perez", "Sanchez", "Ramirez", "Torres", "Flores", "Rivera",
        "Gomez", "Diaz", "Reyes", "Cruz", "Morales", "Ortiz", "Gutierrez", "Vargas",
    ],
    "aian": [
        "Runningwater", "Begay", "Yazzie", "Swiftwind", "Greyhorse", "Eagleheart",
        "Proudfoot", "Raincloud", "Clearwater", "Longbow", "Blackhorse",
        "Redhawk", "Whitecloud", "Ironhorse", "Boldwind", "Deerhunter",
        "Nighthorse", "Wolfrunner", "Strongbow", "Braveheart",
    ],
    "nhpi": [
        "Taufa", "Faleolo", "Tuivaga", "Mataele", "Fifita", "Taufa",
        "Lokeni", "Tuilagi", "Fono", "Malolo", "Fatu", "Leota",
        "Savali", "Simi", "Nonu", "Lilo", "Tanuvasa", "Faleolo",
        "Seumanutafa", "Tulafono",
    ],
    "other": [
        "Cohen", "Okafor", "Petrov", "Nkrumah", "Hassan", "Obi", "Dupont",
        "Kowalski", "Schmidt", "Muller", "Bauer", "Fischer", "Weber",
        "Alonso", "Ferreira", "Nakashima", "Volkov", "Ali", "Mensah", "Dlamini",
    ],
}


def _rac1p_to_group(rac1p: str) -> str:
    """Map RAC1P code to name-list group key."""
    mapping = {
        "1": "white",
        "2": "black",
        "3": "aian",   # American Indian and Alaska Native
        "4": "aian",   # Alaska Native alone
        "5": "aian",   # AIAN combined
        "6": "asian",
        "7": "nhpi",
        "8": "other",
        "9": "other",  # Two or more races
    }
    return mapping.get(str(rac1p), "other")


def _sex_label(sex_code: str) -> str:
    return "Male" if str(sex_code) == "1" else "Female"


def _sex_key(sex_code: str) -> str:
    return "male" if str(sex_code) == "1" else "female"


def _race_ethnicity_label(rac1p: str, hisp: str) -> str:
    """Human-readable race/ethnicity label. Hispanic overrides race.
    HISP=01 means NOT Hispanic; any other value (02-24) means Hispanic."""
    hisp_val = str(hisp).lstrip("0") or "0"
    if hisp_val != "1":
        return "Hispanic/Latino"
    labels = {
        "1": "White",
        "2": "Black or African American",
        "3": "American Indian/Alaska Native",
        "4": "Alaska Native",
        "5": "American Indian/Alaska Native",
        "6": "Asian",
        "7": "Native Hawaiian/Pacific Islander",
        "8": "Other",
        "9": "Two or More Races",
    }
    return labels.get(str(rac1p), "Other")


def _generate_name(agent_id: int, rac1p: str, hisp: str, sex_code: str) -> str:
    """Pick a name deterministically using per-agent RNG. REQ-007."""
    rng = random.Random(agent_id)
    # Hispanic HISP != 01 overrides race group (HISP=01 means not Hispanic)
    hisp_val = str(hisp).lstrip("0") or "0"
    group = "hispanic" if hisp_val != "1" else _rac1p_to_group(rac1p)
    sex_key = _sex_key(sex_code)

    first_list = FIRST_NAMES.get((group, sex_key)) or FIRST_NAMES[("other", sex_key)]
    last_list = LAST_NAMES.get(group) or LAST_NAMES["other"]

    first = rng.choice(first_list)
    last = rng.choice(last_list)
    return f"{first} {last}"


def _map_occupation(occp: str) -> str:
    """Resolve OCCP code to human-readable occupation. REQ-006."""
    if occp is None or occp == "":
        return "Not in labor force"
    occp_str = str(occp).strip()
    if occp_str in ("N", "None", ""):
        return "Not in labor force"
    if occp_str in OCCP_LOOKUP:
        return OCCP_LOOKUP[occp_str]
    # Range-based fallback
    try:
        code_int = int(occp_str)
    except ValueError:
        return "Other"
    for lo, hi, label in OCCP_RANGES:
        if lo <= code_int <= hi:
            return label
    return "Other"


def _is_swe(occp: str) -> bool:
    """Return True if this OCCP code counts against the software-engineer cap."""
    return str(occp).strip() in SWE_CODES


def _tenure_label(ten: str) -> str:
    """TEN codes 1,2 = Owner; 3,4 = Renter."""
    return "Owner" if str(ten) in ("1", "2") else "Renter"


def _income_bracket(hincp: int) -> str:
    """Derive household_income_bracket from HINCP. REQ-010."""
    if hincp < 35_000:
        return "<$35k"
    if hincp < 75_000:
        return "$35k-$75k"
    if hincp < 120_000:
        return "$75k-$120k"
    return ">$120k"


# Occupation category -> (O, C, E, A, N) adjustments. Checked in order, first
# match wins, so more specific categories (tech, healthcare) precede broader
# ones (business, office). Keywords cover both specific titles and the broad
# group labels the OCCP lookup emits (e.g. "Management", "Office/Admin").
_OCC_CATEGORIES = (
    ("tech/engineering",
     ("software", "developer", "engineer", "computer", "data", "analyst", "web",
      "network", "database", "systems", "aerospace", "programmer", "scientist"),
     (0.8, 0.4, -0.2, -0.2, 0.0)),
    ("healthcare",
     ("nurse", "doctor", "physician", "pharmacist", "therapist", "surgeon",
      "paramedic", "dentist", "medical", "health", "clinical", "phlebotomist"),
     (0.0, 0.6, 0.2, 0.8, 0.0)),
    ("arts/media",
     ("artist", "musician", "photographer", "actor", "dancer", "designer",
      "writer", "editor", "reporter", "producer", "director", "media", "arts"),
     (1.2, -0.3, 0.3, 0.0, 0.3)),
    ("education",
     ("teacher", "professor", "instructor", "librarian", "education", "tutor",
      "teaching", "school"),
     (0.4, 0.5, 0.2, 0.6, 0.0)),
    ("business/mgmt",
     ("management", "manager", "business", "finance", "financial", "sales agent",
      "executive", "accountant", "marketing", "consultant"),
     (0.2, 0.6, 0.5, -0.2, -0.2)),
    ("service/food",
     ("food", "waiter", "server", "bartender", "barista", "chef", "baker",
      "cashier", "counter", "dishwasher", "personal care", "clerk", "retail"),
     (0.0, -0.1, 0.4, 0.5, 0.2)),
    ("office/admin",
     ("office", "admin", "operator", "station", "clerical", "receptionist"),
     (0.0, 0.4, -0.2, 0.0, 0.0)),
)


def _derive_ocean(
    agent_id: int,
    age: int,
    sex_code: str,
    household_income: int,
    tenure: str,
    occupation: str,
) -> dict:
    """
    Derive OCEAN scores from demographic correlates (REQ-008). Demographics are
    the PRIMARY, dominant signal; a small per-agent noise term adds within-group
    variety. Age and household income act continuously; sex, tenure, and
    occupation-category act as signed terms. See the module docstring for
    coefficients and rationale.
    """
    BASE = 5.5
    o = c = e = a = n = BASE

    # Age — continuous "maturity principle" (older => more C/A, less N/O/E).
    age_factor = (age - 45) / 17.0
    o += -0.5 * age_factor
    c += 0.7 * age_factor
    e += -0.3 * age_factor
    a += 0.6 * age_factor
    n += -0.7 * age_factor

    # Sex — replicated population differences in Neuroticism and Agreeableness.
    if str(sex_code) == "2":  # Female
        o += 0.1; c += 0.2; a += 0.6; n += 0.9
    else:  # Male
        a -= 0.2; n -= 0.4

    # Household income — continuous SES on a log scale. Pivot 5.08 ~ $120k,
    # near SF's median household income, so ses=0 is a typical SF household.
    effective_income = max(household_income, 15_000)
    ses = (math.log10(effective_income) - 5.08) / 0.5
    ses = max(-1.5, min(1.8, ses))
    o += 0.3 * ses
    c += 0.5 * ses
    e += 0.3 * ses
    a += -0.1 * ses
    n += -0.5 * ses

    # Tenure — ownership correlates with stability (higher C, lower N).
    if tenure == "Owner":
        c += 0.4; a += 0.2; n -= 0.4
    else:
        o += 0.2; c -= 0.2; e += 0.2; n += 0.3

    # Occupation category — every worker gets a signal; unmatched titles and
    # "not in labor force" contribute nothing.
    occ_lower = occupation.lower()
    for _label, keywords, (do, dc, de, da, dn) in _OCC_CATEGORIES:
        if any(k in occ_lower for k in keywords):
            o += do; c += dc; e += de; a += da; n += dn
            break

    # Per-agent noise — within-group individual variation (secondary to
    # demographics). Deterministic per agent for reproducibility.
    rng = random.Random(f"{agent_id}_ocean")
    o += rng.gauss(0, 0.7)
    c += rng.gauss(0, 0.7)
    e += rng.gauss(0, 0.7)
    a += rng.gauss(0, 0.7)
    n += rng.gauss(0, 0.7)

    def clamp(val: float) -> float:
        return round(max(1.0, min(10.0, val)), 1)

    return {
        "openness": clamp(o),
        "conscientiousness": clamp(c),
        "extraversion": clamp(e),
        "agreeableness": clamp(a),
        "neuroticism": clamp(n),
    }


def _build_profile(
    name: str,
    age: int,
    occupation: str,
    neighborhood: str,
    household_income: int,
    tenure: str,
    ocean: dict,
) -> str:
    """
    Generate a 2-sentence behavioral profile in third-person.
    Deterministic — no LLM call. REQ-009.
    """
    # Sentence 1: Who they are and their life context
    tenure_phrase = "owns their home" if tenure == "Owner" else "rents in the city"
    income_phrase = (
        f"in a household earning ${household_income:,}/year"
        if household_income > 0
        else "with modest household income"
    )
    s1 = (
        f"{name} is a {age}-year-old {occupation} living in {neighborhood}, {tenure_phrase}, "
        f"and {income_phrase}."
    )

    # Sentence 2: Personality-driven behavioral tendencies from OCEAN
    o = ocean["openness"]
    c = ocean["conscientiousness"]
    e = ocean["extraversion"]
    a = ocean["agreeableness"]
    n = ocean["neuroticism"]

    traits = []
    if o >= 7.0:
        traits.append("intellectually curious and open to new ideas")
    elif o <= 4.0:
        traits.append("practical and preferring familiar routines")
    if c >= 7.0:
        traits.append("highly organized and goal-driven")
    elif c <= 4.0:
        traits.append("flexible with a relaxed approach to structure")
    if e >= 7.0:
        traits.append("socially energetic and outgoing")
    elif e <= 4.0:
        traits.append("reserved and comfortable working independently")
    if a >= 7.0:
        traits.append("cooperative and community-minded")
    elif a <= 4.0:
        traits.append("direct and willing to push back on consensus")
    if n >= 7.0:
        traits.append("sensitive to stress and attuned to risks")
    elif n <= 3.5:
        traits.append("emotionally stable and resilient under pressure")

    if not traits:
        traits.append("balanced across personality dimensions")

    if len(traits) == 1:
        trait_str = traits[0]
    elif len(traits) == 2:
        trait_str = f"{traits[0]} and {traits[1]}"
    else:
        trait_str = ", ".join(traits[:-1]) + f", and {traits[-1]}"

    s2 = f"They tend to be {trait_str}."
    return f"{s1} {s2}"


# ---------------------------------------------------------------------------
# Agent memory — 3 past delivery-app experiences (optional, LLM-backed)
# ---------------------------------------------------------------------------

# o4-mini persona prompt for eliciting delivery-app experiences. Uses the
# `developer` role (not `system`) like run_scenario.py. Must NOT mention Prop F,
# Proposition F, 2021, or any real prior vote, and must NOT tell the persona how
# to vote — we only want lived, concrete experiences.
_MEMORY_DEVELOPER_TEMPLATE = """\
You are a {age}-year-old {race_ethnicity} {sex} living in {neighborhood}, \
San Francisco, working as a {occupation}.
Your household earns ${household_income:,}/year and {tenure_desc}.

{profile}

Recall your personal history with food-delivery apps (DoorDash, Uber Eats, \
Grubhub) — as a customer, a gig delivery driver, and/or a small-business or \
restaurant owner, whichever fits this person's life. Respond ONLY with valid JSON.\
"""

_MEMORY_USER_PROMPT = (
    "Describe exactly 3 specific, realistic past experiences this person has had "
    "with food-delivery apps. Each is one concrete first-person sentence grounded "
    "in their life circumstances. Vary the experiences (they need not all be "
    "positive or all negative). Do not reference any ballot measure, election, or "
    "how you would vote.\n\n"
    'Respond in JSON: {"experiences": ["...", "...", "..."]}'
)


def _build_memory_developer_prompt(agent: dict) -> str:
    """Persona prompt used to elicit an agent's delivery-app experiences."""
    tenure_desc = (
        "you own your home" if agent["tenure"] == "Owner" else "you rent your home"
    )
    return _MEMORY_DEVELOPER_TEMPLATE.format(
        age=agent["age"],
        sex=agent["sex"],
        race_ethnicity=agent["race_ethnicity"],
        occupation=agent["occupation"],
        neighborhood=agent["neighborhood"],
        household_income=max(0, agent["household_income"]),
        tenure_desc=tenure_desc,
        profile=agent["profile"],
    )


def _parse_experiences(content: str) -> list:
    """
    Parse the model's JSON for a list of exactly 3 non-empty experience strings.
    Returns the cleaned list, or [] on any malformed response.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    raw = data.get("experiences") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) != DELIVERY_EXPERIENCE_COUNT:
        return []
    cleaned = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    if len(cleaned) != DELIVERY_EXPERIENCE_COUNT:
        return []
    return cleaned


def _generate_delivery_experiences(agent: dict, client) -> list:
    """
    Ask o4-mini for 3 past food-delivery-app experiences for one agent.
    One retry on a malformed response; returns [] if both attempts fail so a
    single bad call never aborts population generation.
    """
    developer_prompt = _build_memory_developer_prompt(agent)
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MEMORY_MODEL,
                reasoning_effort="low",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "developer", "content": developer_prompt},
                    {"role": "user", "content": _MEMORY_USER_PROMPT},
                ],
            )
            experiences = _parse_experiences(response.choices[0].message.content)
            if experiences:
                return experiences
        except Exception as exc:  # network / API error — retry then give up
            print(
                f"  delivery-experience call failed for agent {agent['id']}: {exc}",
                file=sys.stderr,
            )
    return []


def _attach_delivery_experiences(agents: list) -> None:
    """
    Populate agent["delivery_experiences"] for every agent, concurrently.
    Requires OPENAI_API_KEY. Mutates the agents in place.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required to generate agent memory (delivery-app "
            "experiences). Set it or disable the agent-memory option."
        )
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    client = OpenAI()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        results = executor.map(
            lambda agent: _generate_delivery_experiences(agent, client), agents
        )
        for agent, experiences in zip(agents, results):
            agent["delivery_experiences"] = experiences


# ---------------------------------------------------------------------------
# Census data fetch & cache — REQ-001, REQ-002
# ---------------------------------------------------------------------------

def _fetch_pums() -> list[dict]:
    """Fetch Census PUMS data or load from cache. REQ-001, REQ-002."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        print(f"Loading PUMS data from cache: {CACHE_PATH}", file=sys.stderr)
        return _load_cache()

    if not _census_key:
        raise RuntimeError(
            "CENSUS_API_KEY environment variable is required.\n"
            "Get a free key at: https://api.census.gov/data/key_signup.html\n"
            "Then: export CENSUS_API_KEY=your_key_here"
        )

    print("Fetching PUMS data from Census API...", file=sys.stderr)
    try:
        resp = requests.get(CENSUS_URL, params=CENSUS_PARAMS, timeout=180)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Census API request failed: {exc}\n"
            f"URL: {CENSUS_URL}\nParams: {CENSUS_PARAMS}"
        ) from exc

    raw = resp.json()
    if not raw or len(raw) < 2:
        raise RuntimeError("Census API returned empty or malformed response.")

    # raw[0] is the header row; raw[1:] are data rows
    headers = raw[0]
    all_records = [dict(zip(headers, row)) for row in raw[1:]]

    # Filter to SF PUMAs only. PUMA20 values may lack leading zeros.
    records = []
    for r in all_records:
        puma = str(r.get("PUMA20", "")).zfill(5)
        if puma in SF_PUMAS:
            records.append(r)

    print(f"Filtered {len(records)} SF records from {len(all_records)} CA records", file=sys.stderr)

    # Cache SF-only records to CSV
    with open(CACHE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    print(f"Cached {len(records)} records to {CACHE_PATH}", file=sys.stderr)
    return records


def _load_cache() -> list[dict]:
    """Load previously cached CSV. REQ-002."""
    try:
        with open(CACHE_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as exc:
        print(f"Cache read failed ({exc}), re-fetching...", file=sys.stderr)
        CACHE_PATH.unlink(missing_ok=True)
        return _fetch_pums()


# ---------------------------------------------------------------------------
# Representative sampling — REQ-004, REQ-011
# The 30-agent sample is drawn to mirror SF's actual adult population on several
# marginals (sex, race/ethnicity, tenure, household-income bracket). Targets are
# COMPUTED from the PUMS person-weights themselves (self-calibrating), rounded to
# 30 via largest-remainder, then hit with balanced greedy selection. This gives a
# statistically representative sample — real income inequality, the tech/service
# mix, renters vs owners, and ethnic diversity — instead of one lucky draw.
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 30
MAX_SWE = 4  # REQ-011

# Household-income brackets spanning SF's inequality (low to very high).
_HH_INCOME_BRACKETS = (
    ("<$30k", 30_000),
    ("$30-75k", 75_000),
    ("$75-150k", 150_000),
    ("$150-300k", 300_000),
    ("$300k+", float("inf")),
)

# Race/ethnicity buckets used as sampling targets; rare categories fold to "Other".
_RACE_TARGETS = ("White", "Asian", "Hispanic/Latino",
                 "Black or African American", "Two or More Races", "Other")


def _hh_bracket(hincp: int) -> str:
    """Map a household income to its bracket label."""
    for label, upper in _HH_INCOME_BRACKETS:
        if hincp < upper:
            return label
    return _HH_INCOME_BRACKETS[-1][0]


def _sample_categories(rec: dict) -> dict:
    """Extract the sampling-dimension category values for a raw PUMS record."""
    race = _race_ethnicity_label(str(rec.get("RAC1P", "")), str(rec.get("HISP", "1")))
    if race not in _RACE_TARGETS:
        race = "Other"
    try:
        hincp = int(rec.get("HINCP", 0) or 0)
    except (ValueError, TypeError):
        hincp = 0
    return {
        "sex": "Female" if str(rec.get("SEX")) == "2" else "Male",
        "race": race,
        "tenure": "Owner" if str(rec.get("TEN")) in ("1", "2") else "Renter",
        "income": _hh_bracket(hincp),
    }


def _largest_remainder(weighted: dict, total: int) -> dict:
    """Round weighted category shares to integer counts summing exactly to total."""
    grand = sum(weighted.values()) or 1
    exact = {k: total * v / grand for k, v in weighted.items()}
    counts = {k: int(v) for k, v in exact.items()}
    leftover = total - sum(counts.values())
    order = sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)
    for k in order[:leftover]:
        counts[k] += 1
    return counts


def _compute_targets(adults: list) -> dict:
    """Per-dimension integer quotas (each summing to SAMPLE_SIZE) computed from
    the PWGTP-weighted marginals of the adult population."""
    dims = ("sex", "race", "tenure", "income")
    weighted = {d: {} for d in dims}
    for rec in adults:
        w = int(rec["PWGTP"])
        cats = _sample_categories(rec)
        for d in dims:
            weighted[d][cats[d]] = weighted[d].get(cats[d], 0) + w
    return {d: _largest_remainder(weighted[d], SAMPLE_SIZE) for d in dims}


def _balanced_sample(adults: list, seed: int) -> list:
    """Select SAMPLE_SIZE records matching SF marginals on every dimension.

    Targets come from the population's own person-weights. A greedy balanced pass
    repeatedly picks the record that fills the most still-open quota cells, breaking
    ties by a PWGTP-weighted random order (Efraimidis-Spirakis) so within-cell
    choices stay representative. The SWE cap (REQ-011) is enforced throughout.
    """
    rng = random.Random(seed)
    targets = _compute_targets(adults)
    remaining = {d: dict(counts) for d, counts in targets.items()}
    dims = tuple(targets.keys())

    # Weighted-random order: higher PWGTP => key closer to 1 => selected earlier.
    order = sorted(adults, key=lambda r: rng.random() ** (1.0 / max(1, int(r["PWGTP"]))),
                   reverse=True)
    cats = {id(rec): _sample_categories(rec) for rec in order}

    chosen, used, swe = [], set(), 0

    def gain(rec):
        c = cats[id(rec)]
        return sum(1 for d in dims if remaining[d].get(c[d], 0) > 0)

    while len(chosen) < SAMPLE_SIZE and len(used) < len(order):
        best, best_gain = None, -1
        for rec in order:
            if id(rec) in used:
                continue
            if _is_swe(str(rec.get("OCCP", "")).strip()) and swe >= MAX_SWE:
                continue
            g = gain(rec)
            if g > best_gain:
                best, best_gain = rec, g
                if g == len(dims):
                    break
        if best is None:  # only SWE-capped records left; relax cap to reach 30
            best = next(rec for rec in order if id(rec) not in used)
        used.add(id(best))
        chosen.append(best)
        if _is_swe(str(best.get("OCCP", "")).strip()):
            swe += 1
        c = cats[id(best)]
        for d in dims:
            if remaining[d].get(c[d], 0) > 0:
                remaining[d][c[d]] -= 1
    return chosen, targets


# ---------------------------------------------------------------------------
# Core generate function — importable by app.py
# ---------------------------------------------------------------------------

def generate(seed: int = 42, use_memory: bool = False) -> list[dict]:
    """
    Fetch PUMS data, weighted-sample 30 SF adult personas, and return as list[dict].
    REQ-001 through REQ-013. Importable interface for app.py.

    All global randomness seeded with random.seed(seed).
    Per-agent randomness uses random.Random(agent_id) for reproducibility.

    If use_memory is True, each agent additionally gets a "delivery_experiences"
    list of 3 LLM-generated past food-delivery-app experiences (requires
    OPENAI_API_KEY). This is the only code path in this module that calls the LLM.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = _fetch_pums()

    # Filter adults — REQ-003
    adults = [r for r in records if int(r["AGEP"]) >= 18]
    print(f"Adults (AGEP >= 18): {len(adults)}", file=sys.stderr)

    # Representative sampling — REQ-004, REQ-011. Match SF's actual adult
    # marginals (sex, race/ethnicity, tenure, household income) using targets
    # computed from the PUMS person-weights. See _balanced_sample.
    random.seed(seed)
    sampled, targets = _balanced_sample(adults, seed)

    # Report achieved vs target marginals so representativeness is auditable.
    for dim in ("sex", "race", "tenure", "income"):
        got = {}
        for rec in sampled:
            cat = _sample_categories(rec)[dim]
            got[cat] = got.get(cat, 0) + 1
        pairs = ", ".join(f"{k} {got.get(k, 0)}/{v}" for k, v in targets[dim].items())
        print(f"  sample {dim}: {pairs}", file=sys.stderr)

    agents = []
    for i, rec in enumerate(sampled[:30]):
        agent_id = i + 1  # 1-indexed

        age = int(rec["AGEP"])
        sex_code = str(rec.get("SEX", "1"))
        rac1p = str(rec.get("RAC1P", "1"))
        hisp = str(rec.get("HISP", "1"))
        occp = str(rec.get("OCCP", "")).strip()
        ten = str(rec.get("TEN", "3"))
        puma_raw = str(rec.get("PUMA20", rec.get("public use microdata area", rec.get("PUMA", ""))))
        # PUMA code comes back without leading zero from Census API; normalize to 5 digits
        puma = puma_raw.zfill(5)

        try:
            hincp = int(rec.get("HINCP", 0) or 0)
        except (ValueError, TypeError):
            hincp = 0

        # Neighborhood — per-agent RNG for reproducibility
        neighborhood_list = PUMA_NEIGHBORHOODS.get(puma, ["San Francisco"])
        rng = random.Random(agent_id)
        neighborhood = rng.choice(neighborhood_list)

        occupation = _map_occupation(occp)
        tenure = _tenure_label(ten)
        name = _generate_name(agent_id, rac1p, hisp, sex_code)
        ocean = _derive_ocean(agent_id, age, sex_code, hincp, tenure, occupation)
        profile = _build_profile(
            name, age, occupation, neighborhood, hincp, tenure, ocean
        )

        agents.append({
            "id": agent_id,
            "name": name,
            "age": age,
            "sex": _sex_label(sex_code),
            "race_ethnicity": _race_ethnicity_label(rac1p, hisp),
            "neighborhood": neighborhood,
            "occupation": occupation,
            "occupation_code": occp,
            "household_income": hincp,
            "household_income_bracket": _income_bracket(hincp),
            "tenure": tenure,
            "puma": puma,
            "ocean": ocean,
            "profile": profile,
        })

    # Optional agent memory — 3 past delivery-app experiences per agent (LLM).
    if use_memory:
        print(
            f"Generating delivery-app experiences for {len(agents)} agents...",
            file=sys.stderr,
        )
        _attach_delivery_experiences(agents)

    return agents


# ---------------------------------------------------------------------------
# CLI entry point — REQ-013
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 30 synthetic SF residents from ACS 2022 PUMS data."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Give each agent 3 LLM-generated past delivery-app experiences "
        "(requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    agents = generate(seed=args.seed, use_memory=args.memory)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(agents)} agents to {output_path}")
