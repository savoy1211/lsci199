import requests

def download_gutenberg_text(url):
  """ Download a text file from a Project Gutenberg URL and return it as a string. """
  return requests.get(url).content.decode('utf-8', 'ignore')[1:] # Remove the weird initial character

urls = {
    'alice_in_wonderland': ['https://www.gutenberg.org/files/11/11-0.txt'],
    'sherlock_holmes': ['https://www.gutenberg.org/files/1661/1661-0.txt'],
    'frankenstein': ['https://www.gutenberg.org/files/84/84-0.txt'],
    'metamorphosis': ['http://www.gutenberg.org/cache/epub/5200/pg5200.txt'],
    'fairy_tales_grimm': ['https://www.gutenberg.org/files/2591/2591-0.txt'],
    'walden': ['https://www.gutenberg.org/files/205/205-0.txt'],
    'little_women': ['http://www.gutenberg.org/cache/epub/514/pg514.txt'],
    'tom_sawyer': ['https://www.gutenberg.org/files/74/74-0.txt'],
    'war_and_peace': ['https://www.gutenberg.org/files/2600/2600-0.txt'],
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
    'dubliners': ['https://www.gutenberg.org/files/2814/2814-0.txt'],
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
    "call_of_wild": ['https://www.gutenberg.org/files/215/215-0.txt'],
    'study_in_scarlet': ['https://www.gutenberg.org/files/244/244-0.txt'],
    'happy_prince': ['https://www.gutenberg.org/files/902/902-0.txt'],
    'good_and_evil': ['http://www.gutenberg.org/cache/epub/4363/pg4363.txt'],
    'sign_of_four': ['https://www.gutenberg.org/files/2097/2097-0.txt'],
    'benjamin_franklin': ['http://www.gutenberg.org/cache/epub/148/pg148.txt'],
    'sense_and_sensibility': ['http://www.gutenberg.org/cache/epub/161/pg161.txt'],
    'man_thinketh': ['http://www.gutenberg.org/cache/epub/4507/pg4507.txt'],
    'common_sense': ['http://www.gutenberg.org/cache/epub/147/pg147.txt'],
    'anthem': ['https://www.gutenberg.org/files/1250/1250-0.txt'],
    'shakespeare': ['https://www.gutenberg.org/files/100/100-0.txt'],
    'ulysses_grant': ['http://www.gutenberg.org/cache/epub/4367/pg4367.txt'],
    'candide': ['http://www.gutenberg.org/cache/epub/19942/pg19942.txt'],
    'problems_of_philosophy': ['http://www.gutenberg.org/cache/epub/5827/pg5827.txt'],
    'kama_sutra': ['http://www.gutenberg.org/cache/epub/27827/pg27827.txt'],
    'the_awakening': ['https://www.gutenberg.org/files/160/160-0.txt'],
    'treasure_island': ['https://www.gutenberg.org/files/120/120-0.txt'],
    'jane_eyre': ['http://www.gutenberg.org/cache/epub/1260/pg1260.txt'],
    'looking_glass': ['https://www.gutenberg.org/files/12/12-0.txt'],
    'the_prince': ['http://www.gutenberg.org/cache/epub/1232/pg1232.txt'],
    'love_and_friendship': ['https://www.gutenberg.org/files/1212/1212-0.txt'],
    'persuasion': ['http://www.gutenberg.org/cache/epub/105/pg105.txt'],
    'lady_susan': ['http://www.gutenberg.org/cache/epub/946/pg946.txt'],
    'mansfield_park': ['https://www.gutenberg.org/files/141/141-0.txt'],
    'northanger_abbey': ['https://www.gutenberg.org/files/121/121-0.txt'],
    'american_claimant': ['https://www.gutenberg.org/files/3179/3179-0.txt'],
    'christian_science': ['https://www.gutenberg.org/files/3187/3187-0.txt'],
    'gondour': ['https://www.gutenberg.org/files/3192/3192-0.txt'],
    'gilded_age': ['https://www.gutenberg.org/files/3178/3178-0.txt'],
    'innocents_abroad': ['https://www.gutenberg.org/files/3176/3176-0.txt'],
    'mississippi': ['https://www.gutenberg.org/files/245/245-0.txt'],
    '30000_bequest': ['https://www.gutenberg.org/files/142/142-0.txt'],
    'alonzo_fitz': ['https://www.gutenberg.org/files/3184/3184-0.txt'],
    'ideal_husband': ['https://www.gutenberg.org/files/885/885-0.txt'],
    'house_of_pomegranates': ['https://www.gutenberg.org/files/873/873-0.txt'],
    'intentions': ['https://www.gutenberg.org/files/887/887-0.txt'],
    'no_importance': ['https://www.gutenberg.org/files/854/854-0.txt'],
    'the_idiot': ['https://www.gutenberg.org/files/2638/2638-0.txt'],
    'poor_folk': ['https://www.gutenberg.org/files/2302/2302-0.txt'],
    'the_possessed': ['https://www.gutenberg.org/files/8117/8117-0.txt'],
    'master_and_man': ['https://www.gutenberg.org/files/986/986-0.txt'],
    'resurrection': ['https://www.gutenberg.org/files/1938/1938-0.txt'],
    'men_live_by': ['https://www.gutenberg.org/files/6157/6157-0.txt'],
    'childhood': ['https://www.gutenberg.org/files/2142/2142-0.txt'],
    'awakening': ['http://www.gutenberg.org/cache/epub/17352/pg17352.txt'],
    'last_bow': ['https://www.gutenberg.org/files/2350/2350-0.txt'],
    'hound_baskervilles': ['http://www.gutenberg.org/cache/epub/3070/pg3070.txt'],
    'lost_word': ['http://www.gutenberg.org/cache/epub/139/pg139.txt'],
    'sherlock_holmes_memoirs': ['https://www.gutenberg.org/files/834/834-0.txt'],
    'poison_belt': ['http://www.gutenberg.org/cache/epub/126/pg126.txt'],
    'sherlock_holmes_returns': ['https://www.gutenberg.org/files/221/221-0.txt'],
    'fire_stories': ['https://www.gutenberg.org/files/54109/54109-0.txt'],
    'sir_nigel': ['https://www.gutenberg.org/files/2845/2845-0.txt'],
    'magic_door': ['http://www.gutenberg.org/cache/epub/5317/pg5317.txt'],
    'valley_of_fear': ['http://www.gutenberg.org/cache/epub/3776/pg3776.txt'],
    'vital_message': ['http://www.gutenberg.org/cache/epub/439/pg439.txt'],
    'danger': ['http://www.gutenberg.org/cache/epub/22357/pg22357.txt'],
    'captain_sharkey': ['http://www.gutenberg.org/cache/epub/34627/pg34627.txt'],
    'raffles_haw': ['https://www.gutenberg.org/files/8394/8394-0.txt'],
    'duet': ['https://www.gutenberg.org/files/5260/5260-0.txt'],
    'german_war': ['http://www.gutenberg.org/cache/epub/42127/pg42127.txt'],
    # # 'firm_of_girdlestone': ['http://www.gutenberg.org/cache/epub/13152/pg13152.txt'],
    # # 'brigadier_gerard': ['http://www.gutenberg.org/cache/epub/11247/pg11247.txt'],
    # # 'boer_war': ['https://www.gutenberg.org/files/3069/3069-0.txt'],
    # # 'fairies': ['http://www.gutenberg.org/cache/epub/47506/pg47506.txt'],
    # # 'congo': ['http://www.gutenberg.org/cache/epub/37712/pg37712.txt'],
    
    
    
    
    
    
}

mega_text_100, mega_text_90, mega_text_50, mega_text_10 = '','','',''

i = 0
for k,v in urls.items():
  mega_text_100 += download_gutenberg_text(v[0])
  # print(k, len(download_gutenberg_text(v[0])))
  if i < 90:
    mega_text_90 += download_gutenberg_text(v[0])
  if i < 50:
    mega_text_50 += download_gutenberg_text(v[0])
  if i < 10:
    mega_text_10 += download_gutenberg_text(v[0])
  i+=1
  
