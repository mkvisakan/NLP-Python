import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5,6,7,8,9,10,11], [1,2,3,4,5,6,7,8,9,10,11], 'r-', [1,2,3,4,5,6,7,8,9,10,11], [1, 3, 4, 10, 2, 5, 7, 6, 11, 9, 8], 'b-',[1,2,3,4,5,6,7,8,9,10,11], [1,4,9,5,2,11,3,8,10, 6,7], 'g--', [1,2,3,4,5,6,7,8,9,10,11], [2,6,10,4,1,11,3,9,8,5,7], 'y--')
plt.xlabel("Language")
plt.ylabel("Rank")

x =['en', 'bg', 'sk', 'sr', 'fa', 'hu', 'ro', 'et', 'po', 'sl', 'cs']


plt.xticks([1,2,3,4,5,6,7,8,9,10,11], x)
plt.legend(('Morp Comp : Lemma with max word types', 'Morp Comp : Average word types per lemma analysis', 'K based analysis : word freq', 'K based analysis : lemma freq'), 'upper left', shadow=True, fancybox=True, bbox_to_anchor=(0.45, 0.25))
plt.show()

