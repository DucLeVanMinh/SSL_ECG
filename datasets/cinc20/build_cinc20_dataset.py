import shutil 
import os


if __name__ == "__main__":

  source = '/content/SSL_ECG/datasets/cinc20/Training_2/'
  dest = '/content/SSL_ECG/datasets/cinc20/WFDB/'

  files = os.listdir(source)
  for f in files:
      shutil.move(source+f, dest)
  shutil.rmtree(source)

  source = '/content/SSL_ECG/datasets/cinc20/Training_PTB/'
  files = os.listdir(source)
  for f in files:
      shutil.move(source+f, dest)
  shutil.rmtree(source)

  source = '/content/SSL_ECG/datasets/cinc20/Training_StPetersburg/'
  files = os.listdir(source)
  for f in files:
      shutil.move(source+f, dest)
  shutil.rmtree(source)

  source = '/content/SSL_ECG/datasets/cinc20/Training_WFDB/'
  files = os.listdir(source)
  for f in files:
      shutil.move(source+f, dest)
  shutil.rmtree(source)

  