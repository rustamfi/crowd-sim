"""
SF Crowd Voting Simulator — Population Generator
=================================================
Fetches ACS 2022 5-year PUMS microdata for San Francisco PUMAs 07501-07507,
weighted-samples 30 adult residents, enriches each with neighborhood, occupation,
name, OCEAN personality scores, and a behavioral profile, then writes to
results/agents.json.

REQ-001 through REQ-013.

OCEAN Demographic Adjustment Table
------------------------------------
All traits start at base 5.5. Adjustments below are additive and stack.
Demographics are the PRIMARY signal; noise (gauss 0, 0.4) is SECONDARY.

Condition                  |  O    |  C    |  E    |  A    |  N    | Rationale
---------------------------|-------|-------|-------|-------|-------|----------
Age < 30                   | +0.7  | -0.5  | +0.6  | -0.3  | +0.5  | Youth: higher O/E, lower C
Age 30-50                  |  0    | +0.3  |  0    | +0.2  |  0    | Mid-life: settling, more A/C
Age > 50                   | -0.4  | +0.6  | -0.4  | +0.5  | -0.5  | Maturity: higher C/A, lower N/O/E
Sex = Female               | +0.2  | +0.2  | +0.2  | +0.5  | +0.5  | Population-level tendencies in A and N
Sex = Male                 |  0    |  0    |  0    | -0.2  | -0.3  | Slight inverse of female adjustments
Income > $150k             | +0.4  | +0.6  | +0.4  |  0    | -0.4  | High earners: conscientious, open, lower anxiety
Income < $30k              | -0.2  | -0.2  | -0.2  | +0.2  | +0.6  | Economic stress: higher neuroticism
Tenure = Own               |  0    | +0.4  |  0    |  0    | -0.3  | Stability correlates with C, lower N
Tenure = Rent              | +0.2  | -0.2  | +0.2  |  0    | +0.2  | Flexibility, slightly higher openness
Tech/creative occupation   | +0.8  | +0.3  |  0    | -0.2  |  0    | Tech/creative: high openness
Service/food occupation    |  0    |  0    | +0.5  | +0.6  | +0.2  | Service: high extraversion/agreeableness
Healthcare occupation      |  0    | +0.6  | +0.2  | +0.7  |  0    | Healthcare: high C/A
Arts/media occupation      | +1.0  | -0.3  | +0.3  |  0    | +0.3  | Artists: very high openness
Education occupation       | +0.3  | +0.4  | +0.3  | +0.5  |  0    | Teachers: agreeable, conscientious
"""

import argparse
import csv
import json
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
# Census API — REQ-001
# ---------------------------------------------------------------------------
CENSUS_URL = "https://api.census.gov/data/2022/acs/acs5/pums"
_census_key = os.environ.get("CENSUS_API_KEY", "")
# PUMS API uses PUMA20 as a variable (not a geography filter).
# We fetch all CA records and filter to SF PUMAs in Python.
# 2020 PUMA codes for San Francisco County (FIPS 06075)
SF_PUMAS = {"07507", "07508", "07509", "07510", "07511", "07512", "07513", "07514"}
CENSUS_PARAMS = {
    "get": "AGEP,OCCP,PINCP,HINCP,TEN,RAC1P,HISP,SEX,PWGTP,PUMA20",
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


def _derive_ocean(
    agent_id: int,
    age: int,
    sex_code: str,
    income: int,
    tenure: str,
    occupation: str,
) -> dict:
    """
    Compute OCEAN scores. Demographics are the PRIMARY signal; noise is secondary.
    See module-level docstring for the full adjustment table. REQ-008.
    """
    BASE = 5.5
    # Start with base values
    o, c, e, a, n = BASE, BASE, BASE, BASE, BASE

    # Age adjustments
    if age < 30:
        o += 0.7; c -= 0.5; e += 0.6; a -= 0.3; n += 0.5
    elif age <= 50:
        c += 0.3; a += 0.2
    else:
        o -= 0.4; c += 0.6; e -= 0.4; a += 0.5; n -= 0.5

    # Sex adjustments
    if str(sex_code) == "2":  # Female
        o += 0.2; c += 0.2; e += 0.2; a += 0.5; n += 0.5
    else:  # Male
        a -= 0.2; n -= 0.3

    # Income adjustments
    if income > 150_000:
        o += 0.4; c += 0.6; e += 0.4; n -= 0.4
    elif income < 30_000:
        o -= 0.2; c -= 0.2; e -= 0.2; a += 0.2; n += 0.6

    # Tenure adjustments
    if tenure == "Owner":
        c += 0.4; n -= 0.3
    else:
        o += 0.2; c -= 0.2; e += 0.2; n += 0.2

    # Occupation adjustments
    occ_lower = occupation.lower()
    tech_creative_keywords = ("software", "developer", "engineer", "computer", "data", "analyst",
                               "web", "network", "database", "security", "it manager", "systems")
    service_food_keywords = ("food", "waiter", "server", "bartender", "barista", "chef", "baker",
                              "cashier", "counter", "dishwasher")
    healthcare_keywords = ("nurse", "doctor", "physician", "pharmacist", "therapist", "surgeon",
                            "paramedic", "dentist", "medical", "healthcare", "clinical", "health aide")
    arts_media_keywords = ("artist", "musician", "photographer", "actor", "dancer", "designer",
                            "writer", "editor", "reporter", "producer", "director", "media", "arts")
    education_keywords = ("teacher", "professor", "instructor", "librarian", "education", "tutor",
                           "postsecondary", "elementary", "school", "teaching assistant")

    if any(k in occ_lower for k in tech_creative_keywords):
        o += 0.8; c += 0.3; a -= 0.2
    elif any(k in occ_lower for k in arts_media_keywords):
        o += 1.0; c -= 0.3; e += 0.3; n += 0.3
    elif any(k in occ_lower for k in healthcare_keywords):
        c += 0.6; e += 0.2; a += 0.7
    elif any(k in occ_lower for k in education_keywords):
        o += 0.3; c += 0.4; e += 0.3; a += 0.5
    elif any(k in occ_lower for k in service_food_keywords):
        e += 0.5; a += 0.6; n += 0.2

    # Per-agent noise — minor individual variation (REQ-008)
    rng = random.Random(f"{agent_id}_ocean")
    o += rng.gauss(0, 0.4)
    c += rng.gauss(0, 0.4)
    e += rng.gauss(0, 0.4)
    a += rng.gauss(0, 0.4)
    n += rng.gauss(0, 0.4)

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
    income: int,
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
        f"earning ${income:,}/year" if income > 0 else "with modest personal income"
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
# Core generate function — importable by app.py
# ---------------------------------------------------------------------------

def generate(seed: int = 42) -> list[dict]:
    """
    Fetch PUMS data, weighted-sample 30 SF adult personas, and return as list[dict].
    REQ-001 through REQ-013. Importable interface for app.py.

    All global randomness seeded with random.seed(seed).
    Per-agent randomness uses random.Random(agent_id) for reproducibility.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = _fetch_pums()

    # Filter adults — REQ-003
    adults = [r for r in records if int(r["AGEP"]) >= 18]
    print(f"Adults (AGEP >= 18): {len(adults)}", file=sys.stderr)

    # Weighted sampling — REQ-004
    weights = [int(r["PWGTP"]) for r in adults]
    random.seed(seed)
    # We may need to sample more than 30 to handle the SWE cap (REQ-011).
    # Strategy: sample a pool of up to 90 candidates, then enforce the cap.
    pool_size = min(90, len(adults))
    pool = random.choices(adults, weights=weights, k=pool_size)

    # REQ-011: Enforce max 4 software engineers.
    # Walk through the pool in order, accepting agents until we have 30,
    # skipping SWE candidates once we've accepted 4.
    swe_count = 0
    MAX_SWE = 4
    sampled = []
    for rec in pool:
        if len(sampled) >= 30:
            break
        occp = str(rec.get("OCCP", "")).strip()
        if _is_swe(occp):
            if swe_count >= MAX_SWE:
                # Skip this record — cap enforced
                continue
            swe_count += 1
        sampled.append(rec)

    # If pool was insufficient (unlikely), pad with non-SWE from adults
    if len(sampled) < 30:
        # Fallback: add more from adults, excluding SWEs if already at cap
        for rec in adults:
            if len(sampled) >= 30:
                break
            if rec in sampled:
                continue
            occp = str(rec.get("OCCP", "")).strip()
            if _is_swe(occp) and swe_count >= MAX_SWE:
                continue
            if _is_swe(occp):
                swe_count += 1
            sampled.append(rec)

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
            pincp = int(rec.get("PINCP", 0) or 0)
        except (ValueError, TypeError):
            pincp = 0
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
        ocean = _derive_ocean(agent_id, age, sex_code, pincp, tenure, occupation)
        profile = _build_profile(
            name, age, occupation, neighborhood, pincp, hincp, tenure, ocean
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
            "income_annual": pincp,
            "household_income": hincp,
            "household_income_bracket": _income_bracket(hincp),
            "tenure": tenure,
            "puma": puma,
            "ocean": ocean,
            "profile": profile,
        })

    # REQ-012: Income spread warning — do not force, only warn
    low_income = sum(1 for a in agents if a["income_annual"] < 35_000)
    high_income = sum(1 for a in agents if a["income_annual"] > 120_000)
    if low_income < 5:
        print(
            f"WARNING (REQ-012): Only {low_income} agents have PINCP < $35k "
            "(expected >= 5). Income spread may be insufficient.",
            file=sys.stderr,
        )
    if high_income < 5:
        print(
            f"WARNING (REQ-012): Only {high_income} agents have PINCP > $120k "
            "(expected >= 5). Income spread may be insufficient.",
            file=sys.stderr,
        )

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
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    agents = generate(seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(agents)} agents to {output_path}")
