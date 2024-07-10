import ollama

user_input = input("What is your question about this profile? ")

response = ollama.chat(model='gemma', messages=[
  {
    'role': 'user',
    'content': user_input + ' of this candidate : ' + """
    
    ROUANI AZEDINE
ANDROID Engineer & Mentor

EXPERIENCE

MAIN SKILLS

Android Engineer (LaPoste)

Experienced Kotlin, Java

CASABLANCA, MAROC 2021 - CURRENT (3 YEARS)

Familiar Swift, Python, Spring, Django

Make delivery apps for LaPoste with Atos
(100k downloads)
Stack : Kotlin, Espresso, Room, Retrofit,
MVVM, Couroutines, Workers, Flow, LiveData,
FireBase, Modularity

Async Coroutines, Flow, Workers,
Livedata, RxJava
Patterns MVVM, MVP, MVC VCS
GIT Github, Gitlab

Android IOS Engineer (Tessa)

DI Hilt / HTTP OkHttp, Retrofit

NICE, FRANCE 2019 - 2021 (2 YEARS)

DataBase Room, SQLite, Realm

Context : Android & IOS application for
taxi companies (100k downloads)
Achievement : Upgrade modern design
and latest Android 10 features
Convert the current Android app into
IOS
Stack : Android Kotlin, IOS Swift, Bluetooth
Communication, Threading

Android Engineer (NouvoNivo)
NICE, FRANCE 2018 - 2019 (1 YEAR)
Context : Instagram Like App with
Psychological features (1M fundings)
Achievement : Develop the app features
according to Silicon Valley investor
agenda, add stories, offline mode...
Stack : Kotlin Native, GraphQl Client,
RxJava, Espresso

EDUCATION
Master 2 Mobile IoT (2018)
Avignon University France

Master 1 Mobile Developer (2017)
Toulon University France

B.S Computer Science Mobile
(2015)
Toulon University France

CONTACT
phone : +212 7 00 76 37 42
mail : azedine.rouani@gmail.com

""",
  },
])
print(response['message']['content'])