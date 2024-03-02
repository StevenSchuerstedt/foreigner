composers = ['bach',
              'beethoven',
                 'chopin',
                   'grieg',
                     'haydn',
                       'liszt',
                         'mendelssohn',
                           'rachmaninov'
                          ]


data_list = {'bach': 
 {'bach': 99, 'beethoven': 0, 'chopin': 0, 'grieg': 1, 'haydn': 0, 'liszt': 0, 'mendelssohn': 0, 'rachmaninov': 0},
   'beethoven':
     {'bach': 0, 'beethoven': 62, 'chopin': 3, 'grieg': 14, 'haydn': 1, 'liszt': 13, 'mendelssohn': 7, 'rachmaninov': 0},
       'chopin':
         {'bach': 0, 'beethoven': 11, 'chopin': 47, 'grieg': 19, 'haydn': 0, 'liszt': 21, 'mendelssohn': 2, 'rachmaninov': 0},
           'grieg':
             {'bach': 0, 'beethoven': 0, 'chopin': 4, 'grieg': 80, 'haydn': 2, 'liszt': 6, 'mendelssohn': 6, 'rachmaninov': 2},
               'haydn':
                 {'bach': 0, 'beethoven': 4, 'chopin': 0, 'grieg': 3, 'haydn': 88, 'liszt': 1, 'mendelssohn': 4, 'rachmaninov': 0},
                   'liszt':
                     {'bach': 0, 'beethoven': 2, 'chopin': 9, 'grieg': 2, 'haydn': 0, 'liszt': 76, 'mendelssohn': 10, 'rachmaninov': 1},
                       'mendelssohn':
                         {'bach': 0, 'beethoven': 8, 'chopin': 0, 'grieg': 3, 'haydn': 0, 'liszt': 6, 'mendelssohn': 83, 'rachmaninov': 0},
                           'rachmaninov':
                             {'bach': 0, 'beethoven': 2, 'chopin': 3, 'grieg': 5, 'haydn': 1, 'liszt': 9, 'mendelssohn': 1, 'rachmaninov': 79}}



#print(data_list['chopin']['liszt'])

print(data_list['beethoven'])

mgenauigkeit = 0
mpräzision = 0
mrückruf = 0
mf1 = 0

for composer in composers:
    print(composer)
    TP = data_list[composer][composer]
    FN = 0
    for c in composers:
        if c == composer:
            continue
        FN += data_list[composer][c]
    FP = 0
    for c in composers:
        if c == composer:
            continue
        FP += data_list[c][composer]
    
    TN = 800 - TP - FP - FN

    genauigkeit = (TP + TN) / (TP + FP + TN + FN)
    präzision = TP / (TP + FP)
    rückruf = TP / (TP + FN)
    f1 = (2*präzision * rückruf)/(präzision + rückruf)

    mgenauigkeit += genauigkeit
    mpräzision += präzision
    mrückruf += rückruf
    mf1 += f1

    print("genauigkeit: ", genauigkeit)
    print("präzision: ", präzision)
    print("rückruf: ", rückruf)
    print("f1: ", f1)

print("mgenauigkeit: ", mgenauigkeit/8)
print("mpräzision: ", mpräzision/8)
print("mrückruf: ", mrückruf/8)
print("mf1: ", mf1/8)
# Genauigkeit = \frac{TP + TN}{TP + FP + TN + FN}

# Praezision = \frac{TP}{TP + FP}

# Rueckruf = \frac{TP}{TP + FN}


# F1 = \frac{2 * Praezision * Rueckruf}{Praezision + Rueckruf}