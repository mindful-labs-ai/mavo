## remove line with whisperx in requirements.freeze2.txt
sed -i '/whisperx/d' devenv/requirements.freeze2.txt

WHISPERX_DIR=file:///home/gq/workspace/simri/prac/whisperx_prac/whisperx2
# pip install --upgrade --force-reinstall -e $WHISPERX_DIR
pip install -e $WHISPERX_DIR

# pip install git+https://github.com/sikbrad/whisperx-sik.git@main#egg=whisperx

# pip freeze \
#   | sed -e 's#-e git+https://github.com/m-bain/whisperX.git@[^#]\+#-e /home/gq/workspace/simri/prac/whisperx_prac/whisperx2#' \
#   > requirements.freeze2.txt

# pip freeze \
#   | sed -e 's#-e git+https://github.com/m-bain/whisperX\.git@[^ ]*#-e /home/gq/workspace/simri/prac/whisperx_prac/whisperx2#' \
#   > requirements.freeze2.txt

# pip freeze > requirements.freeze2.txt