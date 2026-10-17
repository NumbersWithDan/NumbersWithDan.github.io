import struct, pathlib, pandas as pd, numpy as np

def read_dbf(path):
    with open(path,'rb') as f:
        h=f.read(32); n,hl,rl=struct.unpack_from('<IHH',h,4); fields=[]
        for _ in range((hl-33)//32):
            b=f.read(32); fields.append((b[:11].split(b'\0')[0].decode(),chr(b[11]),b[16]))
        f.seek(hl); rows=[]
        for _ in range(n):
            b=f.read(rl)
            if b[:1]==b'*': raise ValueError('Deleted DBF record: geometry alignment must be handled')
            pos=1; row={}
            for name,kind,size in fields:
                s=b[pos:pos+size].decode('utf8',errors='replace').strip();pos+=size
                row[name]=float(s) if kind in 'NF' and s and '*' not in s else (np.nan if kind in 'NF' else s)
            rows.append(row)
    return pd.DataFrame(rows)

def read_points(path):
    points=[]
    with open(path,'rb') as f:
        f.seek(100)
        while head:=f.read(8):
            _,words=struct.unpack('>ii',head); b=f.read(words*2)
            if struct.unpack_from('<i',b)[0]!=1: raise ValueError('Expected Point shape')
            points.append(struct.unpack_from('<dd',b,4))
    return np.asarray(points)

if __name__=='__main__':
    root=pathlib.Path(r'C:\Users\danie\CSML\Project\Data')
    df=read_dbf(root/'uhi_with_all_bands_and_ejscreen_features.dbf')
    print(df.describe().to_string()); print('columns',df.columns.tolist())
    print('raw',read_dbf(root/'UHI/Shape files/af_trav.dbf').describe().to_string())
