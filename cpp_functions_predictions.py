# -*- coding: utf-8 -*-
"""Cpp_functions_predictions.ipynb

# Introduction
In this project I will use bigcode/starcoder2-3b for C++ code completion. As I have some experience in competitive programming, I will try to measure the accuracy of predicting correctly the body of some algorithm related functions (such as bfs, dijkstra, turbomatching).

# Creating dataframe of my previous C++ code
I've selected 44 cpp files each implementing different algorithm. I've created a list, which every entry is as follows:

[name of a file, name of a function implemented in the file, which body I want the AI model to predict].
"""

files = [
    ['2sat_on_square.cpp', 'inline void TWO_SAT(int n) {'],
    ['autobus.cpp', 'inline void dijkstra(int v, int t) {'],
    ['beginner260.cpp', 'inline void add(int p, int q) {'],
    ['bez.cpp', 'inline void compute_sieve() {'],
    ['bieg_na_orientacje.cpp', 'for (int i = 1; i < n; i++) {'],
    ['bipartite.cpp', 'inline void bfs(int v) {'],
    ['brute.cpp', 'inline void print() {'],
    ['centroids3.cpp', 'inline void centroids(int v, int p) {'],
    ['combi1.cpp', 'inline LL fast_powering(LL n, LL k) {'],
    ['constructing0.cpp', 'inline void clear(int n) {'],
    ['dijkstra0.cpp', 'inline LL ceil(LL a, LL b) {'],
    ['dp1.cpp', 'inline LL binomial(int n, int k) {'],
    ['drzewo_stale.cpp', 'inline void add(int x, int t) {'],
    ['drzewo_prze_przed_na_drzewie.cpp', 'inline void add(LL v, LL p, LL q, LL a, LL b, LL x) {'],
    ['dsu.cpp', 'MyInt find(MyInt x, vector<MyInt> &parents) {'],
    ['dsu.cpp', 'inline void make_union(MyInt x, MyInt y, vector<MyInt> &parents) {'],
    ['dsu.cpp', 'inline MyInt number_of_connected_components(vector<MyInt> &parents) {'],
    ['euclidean_cycle.cpp', 'inline bool find_euclidean_path(int v, int n, int m) {'],
    ['extended_euclid_algo.cpp', 'inline LL extended_gcd(LL a, LL b, LL *x, LL *y) { // return gcd'],
    ['gen.cpp', 'long long int rand(long long int a, long long int b) {'],
    ['hashing.cpp', 'bool are_two_suffixes_equal(int a, int b, int length) {'],
    ['hashing.cpp', 'LL get_suffix_hash(int a, int length) {'],
    ['hashing.cpp', 'void calculate_hash(string s, LL power) {'],
    ['hurtownia_19oi.cpp', 'inline LL read(LL v, LL a, LL b, LL p, LL k) {'],
    ['jaskinia.cpp', 'inline int NextInt() { //use getchar_unlocked()'],
    ['kam.cpp', 'inline void push(LL l, LL r, LL v) {'],
    ['kupno_gruntu_15oi.cpp', 'inline LL pole(int x1, int y1, int x2, int y2) {'],
    ['lampy_sloneczne_21oi.cpp', 'inline LL iloczyn_wektorowy(const pair<LL, LL> &a, const pair<LL, LL> &b) {'],
    ['lca.cpp', 'inline LL LCA(LL a, LL b) {'],
    ['macierze0.cpp', 'inline void multiply(LL a[N][N], LL b[N][N]) {'],
    ['newtons_binomial.cpp', 'inline LL inverse_modulo(int x) {'],
    ['okresowosc_18oi.cpp', 'inline void count_KMP(int p, int q) {'],
    ['phi_function.cpp', 'inline void count_phi(int n) {'],
    ['phi_function.cpp', 'inline LL lcm(LL a, LL b) {'],
    ['piloci.cpp', 'inline void add(int v, int x) {'],
    ['plecak.cpp', 'for (LL i = 1; i <= 8; i++) {'],
    ['profesor_szu.cpp', 'inline void toposort(int n) {'],
    ['rob.cpp', 'friend inline Vect operator+(const Vect &a, const Vect &b) {'],
    ['rozliczenia.cpp', 'long long suma(int i) {'],
    ['segment_tree_beats.cpp', 'void update(LL v, LL a, LL b, LL p, LL q, LL x) {'],
    ['swi.cpp', 'inline bool is_prime(LL x) {'],
    ['ton6.cpp', 'inline void calculate_mex(int n) {'],
    ['trie.cpp', 'inline TrieNode *push(char z) {'],
    ['turbo_matching.cpp', 'inline void turbo_matching(short n) {'],
    ['zasada_wlaczen_i_wylaczen.cpp', 'inline void count_mobius_function(LL n) {'],
]
print(len(files))

"""Now I will prepare a pandas dataframe with each row containing:
- file name
- name of the function to implement
- the suffix of the file up to the name of the function
- the body of the function to implement
- the suffix of the file after the function
"""

import pandas as pd

df = pd.DataFrame()
df['file'] = pd.Series(dtype = 'object')
df['function_name'] = pd.Series(dtype = 'object')
for i, file in enumerate(files):
  df.loc[i, 'file'] = file[0]
  df.loc[i, 'function_name'] = file[1]

df.head()

df['prefix'] = pd.Series(dtype = 'object')
df['middle'] = pd.Series(dtype = 'object')
df['suffix'] = pd.Series(dtype = 'object')

"""Because I've decided only to predict body of the functions, I can simply find the end of function in the file, by looking for a first correct bracketing which starts right after the function name. Following function implements this procedure."""

import re

def find_end_of_function(string, function_header_end):
    curly_brackets_pattern = "(\{|\})"
    positions = [match.start() for match in re.finditer(curly_brackets_pattern, string)]
    i, balance = 0, 1
    while i < len(positions) and positions[i] < function_header_end:
        i += 1
    if i >= len(positions):
        print ("File isn't structured correctly 1")
        return None
    for j in range(i, len(positions)):
        if string[positions[j]] == '{':
            balance += 1
        elif string[positions[j]] == '}':
            balance -= 1
        else:
            print("Wrong character")

        if balance == 0:
            return positions[j]
    print ("File isn't structured correctly 2", len(positions), balance)
    return None

import os

def fill_pref_midd_suff_columns(row):
    with open(folder_path + '/' + row['file'], 'r') as file:
        file_contents = file.read()
        header_end = file_contents.index(row['function_name']) + len(row['function_name'])
        function_end = find_end_of_function(file_contents, header_end)
        row['prefix'] = file_contents[:header_end]
        row['middle'] = file_contents[header_end:function_end]
        row['suffix'] = file_contents[function_end:]
    return row

folder_path = 'cpp_files'

df = df.apply(fill_pref_midd_suff_columns, axis = 1)

df.head()

"""Let's see an example of a code split that we've created."""

print(df.loc[0, 'prefix'])
print('MIDDLE_BEGIN')
print(df.loc[0, 'middle'])
print('MIDDLE_END')
print(df.loc[0, 'suffix'])

"""Finally, let's export the dataframe to csv file."""

output_df = df[['prefix', 'middle', 'suffix']]
output_df.to_csv('code_completion_df.csv')

"""# Predicting the bodies of the functions"""

df['prediction'] = pd.Series(dtype = 'object')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint = 'bigcode/starcoder2-3b'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

def generate_predictions(row):
  middle_tag = '<fim_middle>'
  input_text = '<fim_prefix>' + row['prefix'] + '<fim_suffix>' + row['suffix'] + middle_tag
  inputs = tokenizer(input_text, return_tensors="pt").to(device)
  input_ids = inputs['input_ids']
  attention_mask = inputs['attention_mask']
  outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pad_token_id = tokenizer.eos_token_id,
    max_new_tokens = 500
  )
  answer = tokenizer.decode(outputs[0])
  try:
    row['prediction'] = answer[answer.index(middle_tag) + len(middle_tag) : answer.index("<file_sep>")]
  except ValueError:
    row['prediction'] = answer[answer.index(middle_tag) + len(middle_tag) :]
  return row

df = df.apply(generate_predictions, axis = 1)

predictions_df = df[['prefix', 'middle', 'suffix', 'prediction']]
predictions_df.to_csv('code_compiltions_predictions.csv')

"""# Predictions review

Now let's check how accurate are the predictions made by the model. First, I have manually reviewed the predictions and assigned following labels based on their correctness.
"""

df = pd.read_csv('manually_labeled_predictions.csv')
df.columns.values[0] = 'id'
df.set_index(df.columns[0], inplace = True)

df.head()

"""I've labeled the predictions in a following way:
- if the code is correct I've given it a label 1.
- otherwise, I've given it a label from 0 to 1 (without one) based on how serios the problem was and how easy it would be to fix it.

Therefore, code is correct iff its label is 1.

Secondly, let's see use some automatic metrics to see how well the model performed. The metrics are:
- exact_match
- chrf
- bertscore for c++
- google_bleu
"""

df['exact_match'] = pd.Series(dtype = 'float')
df['chrf'] = pd.Series(dtype = 'float')
df['bertscore'] = pd.Series(dtype = 'float')
df['google_bleu'] = pd.Series(dtype = 'float')

# !pip install evaluate
# !pip install sacrebleu
# !pip install bert_score
# !pip install nltk
import evaluate

exact_match = evaluate.load('exact_match')
chrf = evaluate.load('chrf')
bertscore = evaluate.load('bertscore')
google_bleu = evaluate.load('google_bleu')

def calculate_metrics(row):
  row['exact_match'] = exact_match.compute(predictions = [row['prediction']], references = [row['middle']])['exact_match']
  row['chrf'] = chrf.compute(predictions = [row['prediction']], references = [row['middle']])['score']
  row['bertscore'] = bertscore.compute(predictions = [row['prediction']], references = [row['middle']], lang = 'c++')['precision'][0]
  row['google_bleu'] = google_bleu.compute(predictions = [row['prediction']], references = [row['middle']])['google_bleu']
  return row

df = df.apply(calculate_metrics, axis = 1)

df.head()

conclusions_df = df[['prefix', 'middle', 'suffix', 'prediction', 'manual_label', 'exact_match', 'chrf', 'bertscore', 'google_bleu']]
conclusions_df.to_csv('conclusions_df.csv')

"""# Conclussions"""

df = pd.read_csv('conclusions_df.csv')
df.describe()

df['manual_label'].value_counts()

"""As we can see 15 out of 45 predictions were correct. Considering how complex alogrithmic-related problems and codes may be, that this is not a bad result. However, a great majority of predictions which were assigned label greater or equal than 0.9 often were relatively short.

There was one particularly disturbing prediction by the model. Given x <= N function was supposed to check if number x is a prime number. It was supplied with an array prime[] which contained all primes p such that p * p <= N. The body of the function could be implemented in this way:
"""

print(df.loc[40, 'middle'])

"""However, the model predicted it to be:"""

print(df.loc[40, 'prediction'])

"""This implementation is an utter nonsense. Moreover, it was the only prediction which wanted to add more than 500 new tokens, so it threw ValueError.

This example shows us very explicitly how much work is left in the field of code complition.

Let's see which of the metrics correlates best with the labels assigned manually. To measure the correlation I will calculate mse between manually created labels and values assigned by automatic metrics.
"""

import torch
import torch.nn as nn

manual_label_t = torch.tensor(df['manual_label'].values, dtype = torch.float32)
exact_match_t = torch.tensor(df['exact_match'].values, dtype = torch.float32)
chrf_t = torch.tensor(df['chrf'].values, dtype = torch.float32) / 100.;
bertscore_t = torch.tensor(df['bertscore'].values, dtype = torch.float32)
google_bleu_t = torch.tensor(df['google_bleu'].values, dtype = torch.float32)

mse_loss = nn.MSELoss()

print("MSE manual_label & exact_match", mse_loss(manual_label_t, exact_match_t).item())
print("MSE manual_label & chrf", mse_loss(manual_label_t, chrf_t).item())
print("MSE manual_label & bertscore", mse_loss(manual_label_t, bertscore_t).item())
print("MSE manual_label & google_bleu", mse_loss(manual_label_t, google_bleu_t).item())

"""As we can see, the best metric was chrf. Surprisingly, even though we specified the language to c++ in bertscore (so I was believing that it would measure very accurately) didn't perform well when compared to chrf or google_bleu."""

