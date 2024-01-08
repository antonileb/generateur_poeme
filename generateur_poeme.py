from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model


# Lire le fichier texte 
with open("fleurs_mal.txt", 'r' , encoding = 'utf8') as f:
    lines = f.readlines()

# Ici on recupere l'indexe de la premiere ligne et de la derniere ligne du texte
for idx, line in enumerate(lines):
    if "Charles Baudelaire avait un ami" in line:
        first_line = idx
    if "End of the Project Gutenberg EBook of Les Fleurs du Mal, by Charles Baudelaire" in line:
        last_line = idx

# On delimite le texte a seulement le poeme
lines = lines[first_line:last_line]

# Pour chaque element de notre liste : Conversion en minuscules. Suppression des espaces en début et en fin de la chaîne. Suppression des caractères de soulignement (_).
lines = [l.lower().strip().replace('_', '') for l in lines if len(l) > 1]

# Ici on joint notre liste en une seule grande chaine de caractere 
text = " ".join(lines)

# La fonction set permet de créer un 'ensemble', c'est une collection non ordonnée sans éléments en double
# Ici on recupere tous les elements (lettre, chiffre, ponctuation...) qui se trouve dans la variable text et on la trie avec sorted()
characters = sorted(set(text))

# Ici on recupere le nombre d'elements dans notre ensemble
n_characters = len(characters)

# SEQLEN représente la taille de la séquence de lettres à passer en entrée
SEQLEN = 10
step = 1
input_characters, labels = [], []
# Dans notre liste input_characters on va stocker notre featurs, se sont des string de 10 elements
# Dans labels on va stocker notre target, c'est l'element attendu apres les 10 elements de input_characters
# On parcourt le corpus de texte avec une fenêtre glissante
for i in range(0, len(text) - SEQLEN, step):
    input_characters.append(text[i:i + SEQLEN])
    labels.append(text[i + SEQLEN])

# Encodage caractère -> indice du dictionaire
# Ici on créer des dictionnaire, en cles nous avons les caracteres et en valeurs l'indice. l'indice est donner en fonction de sa place dans le set 'characters'
char2index = dict((c, i) for i, c in enumerate(characters))

# Pareil mais dans l'autre sens 
# Encodage de l'indice vers le caractère (utilisé pour décoder les prédictions du modèle)
index2char = dict((i, c) for i, c in enumerate(characters)) 

# Ici on initialise X. X va etre un array avec 3 dimensions,
# il y aura len(input_characters) listes, dans chaques listes il y aura SEQLEN sous-listes et chaques sous-listes auront n_characters valeurs (true ou false)
# Ici la shape de X est : (146120, 10, 59)
X = np.zeros((len(input_characters), SEQLEN, n_characters), dtype=bool)
# Ici on initialise y. y va etre un array de len(input_characters) listes de 59 valeurs (true ou false)
y = np.zeros((len(input_characters), n_characters), dtype=bool)

# Le but : Chaques listes de X representent une chaine de caractere de 10 caractere
# chaque sous liste represent 1 caractere de la chaine de caractere.
# Chaque valeur de la sous liste est la pour nous dire de quel caractere il s'agit. Avec le True or False


# Ici on remplace les valeurs des sous listes par 1 (True) lorsqu'il s'agit du bon caractere 
for idx_seq, sequence in enumerate(input_characters):
    for position, symbol in enumerate(sequence):
        X[idx_seq, position, char2index[symbol]] = 1
    y[idx_seq, char2index[labels[idx_seq]]] = 1


# On separe notre jeu de données
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# On definit notre model 
model = Sequential()

h_size = 128
# Le premier argument h_size permet de spécifier le nombre de neurones en sortie du RNN et donc la taille du vecteur d'activation en sortie.
# Le paramètre return_sequences=False permet de ne renvoyer que l’activation correspondant au dernier pas de temps de la séquence, plutôt que les activations de toute la séquence.
# L’argument optionnel unroll=True permet simplement d’accélérer les calculs en « déroulant » le réseau récurrent plutôt que d’utiliser une boucle for en interne.
model.add(SimpleRNN(h_size, return_sequences=False, input_shape=(SEQLEN, n_characters), unroll=True))

# On construire notre model 
model.add(Dense(n_characters))
model.add(Activation("softmax"))

# On definit le learning_rate et l'optimize 
learning_rate = 0.001
optim = RMSprop(learning_rate=learning_rate)

model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=['accuracy'])
model.summary()

batch_size = 128
num_epochs = 50
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)

scores_train = model.evaluate(X_train, y_train, verbose=1)
scores_test = model.evaluate(X_test, y_test, verbose=1)
print(f"Performances (apprentissage, {model.metrics_names[1]}) = {scores_train[1]*100:.2f}")
print(f"Performances (validation, {model.metrics_names[1]}) = {scores_test[1]*100:.2f}")

model_name = f"SimpleRNN_{h_size}_{num_epochs}epochs"
model.save(model_name)

# On charge les paramètres du réseau récurrent précédemment entraîné à l’aide de la fonction loadModel 
model = load_model(model_name)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()

idx = 10
# index2char permet de repasser de l'encodage one-hot au caractère du dictionnaire
initial_characters = [index2char[np.argmax(c)] for c in X_train[idx]]
initial_text = "".join(initial_characters)
print(f"La séquence n°{idx} est : '{initial_text}'")


test_sequence = np.zeros((1, SEQLEN, n_characters), dtype=bool)
test_sequence[0] = X_train[idx]
prediction = model.predict(test_sequence)
print(prediction)


def sample(probabilities, temperature=1.0):
    probabilities = np.asarray(probabilities).astype('float64')
    # Modifie la distribution selon la valeur de la température
    probabilities = pow(probabilities, 1.0/temperature)
    probabilities /= np.sum(probabilities)
    # Tire des variables aléatoires selon la distribution multinomiale transformée
    random_values = np.random.multinomial(1, probabilities, 1)
    # Renvoie le symbole échantillonné
    return np.argmax(random_values)

# Longueur du texte à générer (en caractères)
text_length = 300
# Température
temperature  = 1.0


generated_text = initial_text
sequence = np.zeros([1, SEQLEN, n_characters], dtype=bool)

for i in range(text_length):
    last_characters = generated_text[-SEQLEN:]
    # Réinitialise à 0 la séquence
    sequence[:] = 0
    # Encodage one-hot du texte
    for pos, c in enumerate(last_characters):
        sequence[0, pos, char2index[c]] = 1

    # Prédiction du modèle
    probabilities = model.predict(sequence, verbose=False)[0]
    index_of_next_character = sample(probabilities, temperature)
    next_character = index2char[index_of_next_character]
    generated_text += next_character

print("Generated text:")
print(generated_text)