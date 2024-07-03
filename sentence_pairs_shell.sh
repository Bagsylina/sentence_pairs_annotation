ADNOTATE_FILE=$1

(PRODIGY_PORT=6871 PRODIGY_ALLOWED_SESSIONS=sergiu,ana,mircea,stadio,userx python3 -m prodigy sentence_pairs sentence_simplest $ADNOTATE_FILE -F sentence_pairs_simplest.py) & (PRODIGY_PORT=6872 PRODIGY_ALLOWED_SESSIONS=sergiu,ana,mircea,stadio,userx python3 -m prodigy sentence_pairs sentence_fluent $ADNOTATE_FILE -F sentence_pairs_most_fluent.py) & (PRODIGY_PORT=6873 PRODIGY_ALLOWED_SESSIONS=sergiu,ana,mircea,stadio,userx python3 -m prodigy sentence_pairs sentence_sense $ADNOTATE_FILE -F sentence_pairs_most_sense.py)
