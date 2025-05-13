import pyarrow
import pandas as pd
import fastparquet

input_files = ["HOH.txt", "OH1.txt", "OH2.txt"]
chunk_size = 1000
delimiter = ' '
parquet_engine = 'fastparquet'  # using fastparquet since it supports append

for input_file in input_files:
    output_file = input_file.replace(".txt", ".parquet")
    first_chunk = True

    for chunk in pd.read_csv(input_file, 
                             sep=delimiter, 
                             engine='python', 
                             chunksize=chunk_size, 
                             header=0, 
                             skipinitialspace=True):
        if first_chunk:
            chunk.to_parquet(output_file, engine=parquet_engine, index=False)
            first_chunk = False
        else:
            chunk.to_parquet(output_file, engine=parquet_engine, index=False, append=True, compression='snappy')
