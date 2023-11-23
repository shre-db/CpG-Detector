from functools import partial
import torch
import streamlit as st
from utils import CpGPredictor
st.set_page_config(page_title="CpG Detector", initial_sidebar_state='collapsed')


st.markdown('<h1 style="text-align: center;">CpG Detector</h1>', unsafe_allow_html=True)
st.markdown('<h5 style="text-align: center;"><i>Consecutive CG detection in DNA sequences.</i></h2>', unsafe_allow_html=True)

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown("""


Welcome to CpG Detector. This web app allows you to input a DNA sequence of variable length and get instant prediction of CpG dinucleotide count
(Cytosine-Guanine pair(s)) in the DNA sequence. To get started, input a DNA sequence in the text box below. 
""")

st.markdown('')
st.markdown('')

# config
LSTM_HIDDEN = 16
LSTM_LAYER = 3
VOCAB_SIZE = 6
OUTPUT_SIZE = 1

model = CpGPredictor(vocab_size=VOCAB_SIZE, embedding_dim=16, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYER, output_size=OUTPUT_SIZE)
try:
    model.load_state_dict(torch.load('models/model_v2.pt', map_location=torch.device('cpu')))
except RuntimeError:
    st.warning("Could not load the weights!")
else:
    st.success("Model is ready ✔")
    st.warning("""
        Important Note:
               
        This model has been trained on DNA sequences with lengths between 64 and 128. While it may perform well within this range, its performance on sequences outside of this range is uncertain. Please be aware of this limitation when submitting sequences for analysis.
    """, icon="ℹ")

st.markdown('')
seq = st.text_input(label="**Enter a DNA sequence**",placeholder="Example: NCACANNTNCGGAGGCGNA...", max_chars=128)

# sample = 'NGTCNNCCAGTTTGAACNGTAACGACGTACATCGACGNCGGNNNTNCAGGTGCNCTGAANNCAACCTGCNCANCGGTTGNCTCANCATANAANGCNCCANAATGATGTNTNNTATCTGCATCCNCNNA'

# Alphabet helpers   
alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}

for char in seq:
    if char not in dna2int.keys():
        st.warning("""
            Prohibited
                   
            Unknown nucleotide(s) found! Please ensure that the sequence consists only of the standard nucleotides (N, A, C, G, T). Recheck the sequence and remove any non-standard characters before resubmitting.
        """, icon='🚫')
        st.stop()

dna2int.update({"pad": 0})
int2dna.update({0: "<pad>"})



def prepare_seq_for_inference(sequence, max_length=128, padding_value=dna2int['pad']):
    # Pad the sequence if it less than max_length
    if len(sequence) < max_length:
        sequence = sequence + [padding_value] * (max_length - len(sequence))
    
    # Add a batch dimension
    input_tensor = torch.tensor(sequence).long().unsqueeze(0)
    return input_tensor 


st.markdown('')
st.markdown('')
if seq != '':
    dnaseq_to_intseq = partial(map, dna2int.get)
    input_sequence_int = list(dnaseq_to_intseq(seq))
    padded_seq = prepare_seq_for_inference(input_sequence_int)

    if st.button("Predict", help="The model will process the provided sequence and predict frequency of CGs."):
        # Infer
        with torch.no_grad():
            output = model(padded_seq)
        st.markdown('')
        st.markdown('')
        st.markdown(f"##### Raw predicted value")
        st.markdown(f"""
        ```
           {output[0][0]}        
        ```
        """)
        st.markdown(f"The model detected :orange[{int(round(float(output[0][0]), 0))}] CG pair(s) in the given sequence.")
