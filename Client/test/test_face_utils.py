from recognition.face_utils import find_closest_match
import torch

def test_find_closest_match(dummy_embedding):
    db_embeddings = torch.rand(5, 512)
    db_names = ["A", "B", "C", "D", "E"]
    db_ids = ["001", "002", "003", "004", "005"]
    emp_id, name, dist = find_closest_match(dummy_embedding, db_embeddings, db_ids, db_names)
    assert emp_id in db_ids or emp_id == "Unknown"
