import os

filePath = '../dataset'


dict_list = []
for i in os.walk(filePath):
    dict_list.append(i)


# relation = ['brother', 'husband', 'birth', 'murder', 'friend',
#           'soldier', 'father', 'sister', 'death', 'suicide',
#           'childhood', 'family', 'girl', 'mother', 'funeral',
#           'son', 'baby', 'hero', 'children', 'teacher',
#           'daughter']
# other = ['power', 'green', 'home', 'running', 'sick',
#          'cinderella', 'car', 'change', 'together', 'memory',
#          'success', 'identity', 'loss', 'today', 'lost',
#          'football', 'sleep', 'poverty', 'silver', 'world', 
#          'life', 'time', 'pink', 'hunting', 'red', 
#          'warning', 'work', 'money', 'war', 'food', 
#          'swimming', 'house', 'dark', 'remember', 'night', 
#          'school', 'future', 'sometimes', 'hair', 'culture', 
#          'believe', 'mirror']
# emotion = ['innocence', 'evil', 'passion', 'sorrow', 'love', 
#            'freedom', 'fear', 'kiss', 'respect', 'funny', 
#            'sympathy', 'racism', 'peace', 'beauty', 'greed', 
#            'destiny', 'lust', 'laughter', 'happiness', 'truth', 
#            'anger', 'beautiful', 'joy', 'justice', 'heaven', 'hate', 
#            'faith', 'courage', 'dream', 'crazy', 'despair', 
#            'god', 'happy','angel','depression', 'hope', 
#            'trust','thanks']
# location = ['paris', 'chicago', 'america', 'june', 'christmas', 'january']
# art = ['music', 'song', 'poetry', 'dance', 'poem']
# romance = ['romance', 'marriage', 'wedding', 'lonely', 'romantic',  'alone']
# travel = ['travel','graduation', 'sun', 'rain', 'winter', 
#           'weather', 'beach', 'summer', 'spring', 'ocean', 
#           'river', 'water', 'sea','rose', 'moon',
#           'sky', 'city','nature','animal','snake',
#           'rainbow','frog','butterfly','star', 'fire']



person = ['brother', 'husband', 'friend',
          'soldier', 'father', 'sister',
          'childhood', 'family', 'girl', 'mother',
          'son', 'baby', 'hero', 'children', 'teacher',
          'daughter', 'home']

other = ['power', 'green', 'running', 'sick',
         'cinderella', 'car', 'change', 'together', 'memory',
         'success', 'identity',
         'football', 'sleep', 'poverty', 'silver', 
         'pink', 'hunting', 'red', 
         'warning', 'work', 'money', 'war', 'food', 
         'swimming', 'house', 'dark', 'remember', 'night', 
         'school', 'hair', 
         'mirror', 'travel']

emotion = ['innocence', 'evil', 'passion', 'sorrow', 'love', 
           'freedom', 'fear', 'kiss', 'respect', 'funny', 
           'sympathy', 'racism', 'peace', 'beauty', 'greed', 
           'lust', 'laughter', 'happiness', 'truth', 
           'anger', 'beautiful', 'joy', 'justice', 'hate', 
           'faith', 'courage', 'dream', 'crazy', 'despair', 
           'happy', 'depression', 'hope', 'believe', 
           'trust','thanks', 'lonely', 'alone', 'loss', 'lost', ]

time_location = ['paris', 'chicago', 'america', 'june', 'christmas',
                 'january', 'today', 'life', 'time', 'future', 'sometimes']

culture = ['music', 'song', 'poetry', 'dance', 'poem', 'culture']

ceremony = ['romance', 'marriage', 'wedding', 'romantic', 'funeral',
            'birth', 'murder', 'suicide', 'death', 'graduation']

religion = ['god', 'angel', 'heaven', 'destiny',]

nature = ['sun', 'rain', 'winter', 'world', 
          'weather', 'beach', 'summer', 'spring', 'ocean', 
          'river', 'water', 'sea','rose', 'moon',
          'sky', 'city','nature','animal','snake',
          'rainbow','frog','butterfly','star', 'fire']

adjusted_tag_dict={}
for i in person:
    adjusted_tag_dict[i] = 'person'
for i in other:
    adjusted_tag_dict[i] = 'other'
for i in emotion:
    adjusted_tag_dict[i] = 'emotion'
for i in time_location:
    adjusted_tag_dict[i] = 'time_location'
for i in culture:
    adjusted_tag_dict[i] = 'culture'
for i in ceremony:
    adjusted_tag_dict[i] = 'ceremony'
for i in religion:
    adjusted_tag_dict[i] = 'religion'
for i in nature:
    adjusted_tag_dict[i] = 'nature'

print(dict_list)


topic = dict_list[1][1]
address_dict = {}
for i in range(2,len(dict_list)):
    print(dict_list[i])
    address_dict[dict_list[i][0]] = dict_list[i][2]
print(address_dict)


filename_poam = {}
for topic_ in address_dict.keys():
    for i in address_dict[topic_]:
        address = topic_+'/'+i
        f2 = open(address,"r")
        lines = f2.readlines()
        for t in range(len(lines)):
            lines[t] = lines[t].replace("\n", "")
        filename_poam[i] = lines




address_key = list(address_dict.keys())
tag_filename = {}
for i in range(len(address_key)):
    #print(topic[i])
    #print(address_dict[address_key[i]])
    tag_filename[topic[i]] = address_dict[address_key[i]]


filename_tag = {}
for tag in tag_filename.keys():
    for filename in tag_filename[tag]:
        filename_tag[filename] = tag


filename_list = list(filename_tag.keys())

#data = [[filename, poem,tag]]
data = []
for filename in filename_list:
    if filename_tag[filename] in adjusted_tag_dict:
        temp = [filename,filename_tag[filename],filename_poam[filename],adjusted_tag_dict[filename_tag[filename]]]
        data.append(temp)


import csv

header = ['filename', 'original_tag','poem','our_tag']


with open('../dataset/fine_data_tsy.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)