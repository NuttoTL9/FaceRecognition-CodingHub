from facenet_pytorch import MTCNN, InceptionResnetV1
from config import DEVICE

mtcnn = MTCNN(keep_all=True, device=DEVICE, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
