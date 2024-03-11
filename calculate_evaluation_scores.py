composers = ['bach',
              'beethoven',
                 'chopin',
                   'grieg',
                     'haydn',
                       'liszt',
                         'mendelssohn',
                           'rachmaninov'
                          ]

#for fbase = gpt2
# data_list = {'bach': 
#  {'bach': 99, 'beethoven': 0, 'chopin': 0, 'grieg': 1, 'haydn': 0, 'liszt': 0, 'mendelssohn': 0, 'rachmaninov': 0},
#    'beethoven':
#      {'bach': 0, 'beethoven': 62, 'chopin': 3, 'grieg': 14, 'haydn': 1, 'liszt': 13, 'mendelssohn': 7, 'rachmaninov': 0},
#        'chopin':
#          {'bach': 0, 'beethoven': 11, 'chopin': 47, 'grieg': 19, 'haydn': 0, 'liszt': 21, 'mendelssohn': 2, 'rachmaninov': 0},
#            'grieg':
#              {'bach': 0, 'beethoven': 0, 'chopin': 4, 'grieg': 80, 'haydn': 2, 'liszt': 6, 'mendelssohn': 6, 'rachmaninov': 2},
#                'haydn':
#                  {'bach': 0, 'beethoven': 4, 'chopin': 0, 'grieg': 3, 'haydn': 88, 'liszt': 1, 'mendelssohn': 4, 'rachmaninov': 0},
#                    'liszt':
#                      {'bach': 0, 'beethoven': 2, 'chopin': 9, 'grieg': 2, 'haydn': 0, 'liszt': 76, 'mendelssohn': 10, 'rachmaninov': 1},
#                        'mendelssohn':
#                          {'bach': 0, 'beethoven': 8, 'chopin': 0, 'grieg': 3, 'haydn': 0, 'liszt': 6, 'mendelssohn': 83, 'rachmaninov': 0},
#                            'rachmaninov':
#                              {'bach': 0, 'beethoven': 2, 'chopin': 3, 'grieg': 5, 'haydn': 1, 'liszt': 9, 'mendelssohn': 1, 'rachmaninov': 79}}


#for fbase = bert
# data_list = {'bach': {'bach': 98, 'beethoven': 1, 'chopin': 1, 'grieg': 0, 'haydn': 0, 'liszt': 0, 'mendelssohn': 0, 'rachmaninov': 0},
#   'beethoven': {'bach': 2, 'beethoven': 52, 'chopin': 12, 'grieg': 15, 'haydn': 6, 'liszt': 4, 'mendelssohn': 6, 'rachmaninov': 3},
#     'chopin': {'bach': 0, 'beethoven': 13, 'chopin': 49, 'grieg': 12, 'haydn': 2, 'liszt': 13, 'mendelssohn': 5, 'rachmaninov': 6},
#       'grieg': {'bach': 0, 'beethoven': 3, 'chopin': 8, 'grieg': 70, 'haydn': 5, 'liszt': 1, 'mendelssohn': 11, 'rachmaninov': 2},
#         'haydn': {'bach': 0, 'beethoven': 1, 'chopin': 0, 'grieg': 0, 'haydn': 98, 'liszt': 0, 'mendelssohn': 0, 'rachmaninov': 1},
#           'liszt': {'bach': 0, 'beethoven': 10, 'chopin': 15, 'grieg': 3, 'haydn': 0, 'liszt': 55, 'mendelssohn': 5, 'rachmaninov': 12},
#             'mendelssohn': {'bach': 0, 'beethoven': 4, 'chopin': 0, 'grieg': 0, 'haydn': 2, 'liszt': 1, 'mendelssohn': 92, 'rachmaninov': 1},
#               'rachmaninov': {'bach': 0, 'beethoven': 4, 'chopin': 3, 'grieg': 2, 'haydn': 7, 'liszt': 5, 'mendelssohn': 2, 'rachmaninov': 77}}

#base line, just guess
# data_list =  {'bach': {'bach': 133, 'beethoven': 114, 'chopin': 124, 'grieg': 128, 'haydn': 111, 'liszt': 143, 'mendelssohn': 131, 'rachmaninov': 116},
#                'beethoven': {'bach': 110, 'beethoven': 113, 'chopin': 131, 'grieg': 138, 'haydn': 150, 'liszt': 122, 'mendelssohn': 129, 'rachmaninov': 107},
#                  'chopin': {'bach': 118, 'beethoven': 123, 'chopin': 135, 'grieg': 134, 'haydn': 123, 'liszt': 120, 'mendelssohn': 124, 'rachmaninov': 123},
#                    'grieg': {'bach': 127, 'beethoven': 124, 'chopin': 143, 'grieg': 121, 'haydn': 117, 'liszt': 137, 'mendelssohn': 119, 'rachmaninov': 112},
#                      'haydn': {'bach': 129, 'beethoven': 112, 'chopin': 142, 'grieg': 129, 'haydn': 121, 'liszt': 125, 'mendelssohn': 107, 'rachmaninov': 135},
#                        'liszt': {'bach': 131, 'beethoven': 111, 'chopin': 132, 'grieg': 120, 'haydn': 120, 'liszt': 130, 'mendelssohn': 126, 'rachmaninov': 130},
#                          'mendelssohn': {'bach': 120, 'beethoven': 121, 'chopin': 106, 'grieg': 112, 'haydn': 129, 'liszt': 137, 'mendelssohn': 144, 'rachmaninov': 131},
#                            'rachmaninov': {'bach': 126, 'beethoven': 114, 'chopin': 124, 'grieg': 121, 'haydn': 127, 'liszt': 126, 'mendelssohn': 135, 'rachmaninov': 127}}

#base line without linear mapping, fbase = gpt2
data_list = {'bach': {'bach': 77, 'beethoven': 0, 'chopin': 9, 'grieg': 3, 'haydn': 2, 'liszt': 1, 'mendelssohn': 2, 'rachmaninov': 6},
              'beethoven': {'bach': 0, 'beethoven': 22, 'chopin': 31, 'grieg': 17, 'haydn': 9, 'liszt': 6, 'mendelssohn': 5, 'rachmaninov': 10},
                'chopin': {'bach': 2, 'beethoven': 16, 'chopin': 44, 'grieg': 11, 'haydn': 11, 'liszt': 6, 'mendelssohn': 4, 'rachmaninov': 6},
                  'grieg': {'bach': 1, 'beethoven': 25, 'chopin': 27, 'grieg': 7, 'haydn': 15, 'liszt': 8, 'mendelssohn': 4, 'rachmaninov': 13},
                    'haydn': {'bach': 8, 'beethoven': 6, 'chopin': 12, 'grieg': 9, 'haydn': 11, 'liszt': 18, 'mendelssohn': 13, 'rachmaninov': 23},
                      'liszt': {'bach': 3, 'beethoven': 5, 'chopin': 23, 'grieg': 12, 'haydn': 14, 'liszt': 25, 'mendelssohn': 9, 'rachmaninov': 9},
                        'mendelssohn': {'bach': 1, 'beethoven': 9, 'chopin': 3, 'grieg': 9, 'haydn': 24, 'liszt': 20, 'mendelssohn': 18, 'rachmaninov': 16},
                          'rachmaninov': {'bach': 0, 'beethoven': 9, 'chopin': 15, 'grieg': 9, 'haydn': 27, 'liszt': 16, 'mendelssohn': 10, 'rachmaninov': 14}}

#base line without linear mapping, fbase = bert
# data_list = {'bach': {'bach': 82, 'beethoven': 1, 'chopin': 3, 'grieg': 9, 'haydn': 1, 'liszt': 0, 'mendelssohn': 2, 'rachmaninov': 2},
#               'beethoven': {'bach': 8, 'beethoven': 17, 'chopin': 27, 'grieg': 12, 'haydn': 11, 'liszt': 7, 'mendelssohn': 15, 'rachmaninov': 3},
#                 'chopin': {'bach': 8, 'beethoven': 15, 'chopin': 30, 'grieg': 11, 'haydn': 10, 'liszt': 3, 'mendelssohn': 14, 'rachmaninov': 9},
#                   'grieg': {'bach': 6, 'beethoven': 14, 'chopin': 17, 'grieg': 20, 'haydn': 10, 'liszt': 8, 'mendelssohn': 11, 'rachmaninov': 14}, 
#                   'haydn': {'bach': 4, 'beethoven': 13, 'chopin': 2, 'grieg': 5, 'haydn': 19, 'liszt': 25, 'mendelssohn': 13, 'rachmaninov': 19},
#                     'liszt': {'bach': 3, 'beethoven': 10, 'chopin': 28, 'grieg': 13, 'haydn': 13, 'liszt': 19, 'mendelssohn': 3, 'rachmaninov': 11},
#                       'mendelssohn': {'bach': 2, 'beethoven': 13, 'chopin': 6, 'grieg': 12, 'haydn': 33, 'liszt': 10, 'mendelssohn': 17, 'rachmaninov': 7},
#                         'rachmaninov': {'bach': 7, 'beethoven': 9, 'chopin': 11, 'grieg': 11, 'haydn': 21, 'liszt': 15, 'mendelssohn': 18, 'rachmaninov': 8}}

#print(data_list['chopin']['liszt'])

#print(data_list['beethoven'])

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