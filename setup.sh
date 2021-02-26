# FIRST MAKE SURE THE ALIGNPAPER ENVIRONMENT IS INSTALLED AND ACTIVATED
# TO DO THIS:
# conda env create -f environment.yml
# conda activate alignpaper

# INSTALL SPACY MODEL: en_core_web_sm
# python -m spacy download en_core_web_sm
# INSTALL SCISPACY MODEL: en_core_sci_sm
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

# GET EBM-NLP DATA
# wget -O data/ebm_nlp_2_00.tar.gz https://github.com/bepnye/EBM-NLP/raw/master/ebm_nlp_2_00.tar.gz
# tar -xf data/ebm_nlp_2_00.tar.gz -C data/
