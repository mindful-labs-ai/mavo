source devenv/activate.sh

export PYTHONPATH=$PYTHONPATH:$(pwd)
python backend/logic/voice_analysis_test.py
