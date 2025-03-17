import sparse_matrix as sparse
import data_loader as dl

# Load datasets
Articles, Bhv_test, Hstr_test, Bhv_val, Hstr_val = dl.load()
print("Datasets loaded successfully.")

### Run data exploration
print("\nStarting data exploration...")
sparse.data_exploration('data')

### Create sparse matrix
#print("\nCreating sparse matrix...")
#user_item_matrix, user_to_idx, article_to_idx = sparse.create_sparse(
#    'data', Articles, Bhv_test, Hstr_test, Bhv_val, Hstr_val, 'user_item_likes_matrix_all.npz'
#)

# Explore sparse matrix
print("\nExploring sparse matrix...")
sparse.Sparse_exploration('user_item_likes_matrix_all.npz', 'data')

print("\nProcess completed successfully.")