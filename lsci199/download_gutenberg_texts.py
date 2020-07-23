import requests

def download_gutenberg_text(url):
  """ Download a text file from a Project Gutenberg URL and return it as a string. """
  return requests.get(url).content.decode('utf-8')[1:] # Remove the weird initial character

urls = {
    'alice_in_wonderland': ['https://www.gutenberg.org/files/11/11-0.txt'],
    'sherlock_holmes': ['https://www.gutenberg.org/files/1661/1661-0.txt'],
    'frankenstein': ['https://www.gutenberg.org/files/84/84-0.txt'],
    'metamorphosis': ['http://www.gutenberg.org/cache/epub/5200/pg5200.txt'],
    'fairy_tales_grimm': ['https://www.gutenberg.org/files/2591/2591-0.txt'],
    'walden': ['https://www.gutenberg.org/files/205/205-0.txt'],
    'little_women': ['http://www.gutenberg.org/cache/epub/514/pg514.txt'],
    'tom_sawyer': ['https://www.gutenberg.org/files/74/74-0.txt'],
    'war_and_peace' ['https://www.gutenberg.org/files/2600/2600-0.txt'],
    'huckleberry_finn': ['https://www.gutenberg.org/files/76/76-0.txt'],
    'plague_year': ['https://www.gutenberg.org/files/376/376-0.txt'],
    'dorian_gray': ['http://www.gutenberg.org/cache/epub/174/pg174.txt'],
    'ullysses': ['https://www.gutenberg.org/files/4300/4300-0.txt'],
    'dracula': ['http://www.gutenberg.org/cache/epub/345/pg345.txt'],
    'emma': ['https://www.gutenberg.org/files/158/158-0.txt'],
    'republic': ['http://www.gutenberg.org/cache/epub/1497/pg1497.txt'],
    'anne_green_gables': ['https://www.gutenberg.org/files/45/45-0.txt'],
    'illiad': ['https://www.gutenberg.org/files/6130/6130-0.txt'],
    'odyssey': ['https://www.gutenberg.org/files/1727/1727-0.txt'],
    'dubliners' ['https://www.gutenberg.org/files/2814/2814-0.txt'],
    'secret_garden': ['https://www.gutenberg.org/files/113/113-0.txt'],
    'sacred_well': ['https://www.gutenberg.org/files/62702/62702-0.txt'],
    'brothers_karamazov': ['https://www.gutenberg.org/files/28054/28054-0.txt'],
    'les_miserables': ['https://www.gutenberg.org/files/135/135-0.txt'],
    'oliver_twist': ['http://www.gutenberg.org/cache/epub/730/pg730.txt'],
    'wuthering_heights': ['http://www.gutenberg.org/cache/epub/768/pg768.txt'],
    'zarathustra': ['https://www.gutenberg.org/files/1998/1998-0.txt'],
    'don_quixote': ['https://www.gutenberg.org/files/996/996-0.txt'],
    'leviathan': ['http://www.gutenberg.org/cache/epub/3207/pg3207.txt'],
    'meditations': ['http://www.gutenberg.org/cache/epub/2680/pg2680.txt'],
    'anna_karenina': ['https://www.gutenberg.org/files/1399/1399-0.txt'],
    'war_of_worlds': ['https://www.gutenberg.org/files/36/36-0.txt'],
    'origins_of_species': ['http://www.gutenberg.org/cache/epub/1228/pg1228.txt'],
    'pygmalion': ['http://www.gutenberg.org/cache/epub/3825/pg3825.txt'],
    'david_copperfield': ['https://www.gutenberg.org/files/766/766-0.txt'],
    'gullivers_travels': ['https://www.gutenberg.org/files/829/829-0.txt'],
    'frederick_douglass': ['http://www.gutenberg.org/cache/epub/23/pg23.txt'],
    'crime_and_punishment': ['https://www.gutenberg.org/files/2554/2554-0.txt'],
    'christmas_carol': ['https://www.gutenberg.org/files/46/46-0.txt'],
    'wizard_of_oz': ['http://www.gutenberg.org/cache/epub/55/pg55.txt'],
    "call_of_wild": ['https://www.gutenberg.org/files/215/215-0.txt']
    
    
    
    
}