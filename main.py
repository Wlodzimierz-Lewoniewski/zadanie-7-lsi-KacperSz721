import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(text):
    return text.lower().translate(str.maketrans("", "", ".,")).split()

def svd_reduce_matrix(matrix, dim):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    U_dim = U[:, :dim]
    S_dim = np.diag(S[:dim])
    Vt_dim = Vt[:dim, :]
    return np.dot(S_dim, Vt_dim), U_dim, S_dim

def term_matrix(docs, query):
    term_list = set()
    for doc in docs + [query]:
        term_list.update(tokenize(doc))
    term_list = sorted(term_list)
    terms_len = len(term_list)
    docs_len = len(docs)

    matrix_terms = np.zeros((terms_len, docs_len))
    for j, doc in enumerate(docs):
        for term in tokenize(doc):
            matrix_terms[term_list.index(term), j] = 1
    return matrix_terms, term_list

def similarities(docs,query,dim):
    matrix_terms_docs, terms = term_matrix(docs, query)
    reduced_matrix, U_dim, S_dim = svd_reduce_matrix(matrix_terms_docs, dim)

    len_query = np.zeros(len(terms))
    for term in tokenize(query):
        if term in terms:
            len_query[terms.index(term)] = 1

    query_inv = np.linalg.inv(S_dim) @ (U_dim.T @ len_query)
    cos_sim = np.round(cosine_similarity(query_inv.reshape(1, -1), reduced_matrix.T).flatten(),2)
    return cos_sim

def main():
    docs_n = int(input())
    docs_list = [input() for _ in range(docs_n)]
    query = input()
    dim = int(input())
    scores_vector = similarities(docs_list, query, dim)
    print(scores_vector.tolist())

if __name__ == "__main__":
    main()
