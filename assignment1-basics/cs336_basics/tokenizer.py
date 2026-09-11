import regex
import os
from collections import Counter,defaultdict
from collections.abc import Iterable
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(
    input_text:str,
    special_tokens:list[str]
)->list[str]:
    '''
    str->计数bytes
    '''
    assert special_tokens is not None, "special_tokens must be a list, not None"
    if not special_tokens:
        parts=[input_text]
    else:
        st_sorted=sorted(special_tokens,key=len,reverse=True)
        st_escaped=[regex.escape(p) for p in st_sorted]
        st_pattern="("+"|".join(st_escaped)+")"
        parts=regex.split(st_pattern,input_text)

    return parts

def get_counter(
    text_list:list[str],
    special_tokens:list[str],
)->Counter[bytes,int]:
    counter=Counter()
    st_set=set(special_tokens)
    for part in text_list:
        if not part or part in st_set:
            continue
        else:
            for match in regex.finditer(PAT,part):
                counter[match.group().encode("utf-8")]+=1

    return counter

def pre_process(
    input_path:str|os.PathLike,
    special_tokens:list[str],
    num_process:int,
    split_special_token:bytes
)->Counter[bytes,int]:
    '''
    对文本中切分出来的字词先进行初始计数
    '''
    counter=Counter()
    with open(input_path,'rb') as f:
        if num_process>1:
            boundaries=find_chunk_boundaries(f,num_process,split_special_token)

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                #chunk是字符串
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                counter+=get_counter(pre_tokenize(chunk,special_tokens),special_tokens)

        else:
            text=f.read().decode("utf-8",errors="ignore")
            counter=get_counter(pre_tokenize(text,special_tokens),special_tokens)
            
    return counter

def merge(
    top_pair:tuple[bytes,bytes],
    new_byte:bytes,
    counter:Counter[tuple[bytes,...],int],
    pair_counter:Counter[tuple[bytes,bytes],int],
    bytes_dict:defaultdict[tuple[bytes,bytes],list[tuple[bytes,...]]]
):
    top_list=list(bytes_dict[top_pair])
    del bytes_dict[top_pair]
    #遍历top_list中的每个part
    for part in top_list:
        if part not in counter:
            continue
        new_tuple=[]
        i=0
        pos=[]
        new_pos=[]
        while i<len(part):
            if i<len(part)-1 and (part[i],part[i+1])==top_pair:
                new_tuple.append(new_byte)
                pos.append(i)
                i+=2
                new_pos.append(len(new_tuple)-1)
            else:
                new_tuple.append(part[i])
                i+=1

        count=counter[part]
        del counter[part]
        new_part=tuple(new_tuple)
        counter[new_part]+=count

        for p in pos:
            #减少旧计数
            if p>0:
                pair_counter[(part[p-1],part[p])]-=count
            pair_counter[top_pair]-=count
            if p<len(part)-2:
                pair_counter[(part[p+1],part[p+2])]-=count
            #增加新计数
        for np in new_pos:
            if np>0:
                pair_counter[(new_part[np-1],new_byte)]+=count
            if np<len(new_part)-1:
                pair_counter[(new_byte,new_part[np+1])]+=count

        for pair in zip(new_part[:-1],new_part[1:]):
            bytes_dict[pair].append(new_part)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merge_rules=[]
    vocab=[]
    if not special_tokens:
        special_tokens=[]
    counter=pre_process(
        input_path=input_path,
        special_tokens=special_tokens,
        num_process=1,
        split_special_token=b"<|endoftext|>")
    
    iter_times=vocab_size-256-len(special_tokens)

    #初步建立词汇表
    vocab.extend(bytes([i]) for i in range(256))
    for st in special_tokens:
        b_st=st.encode("utf-8")
        vocab.append(b_st)

    counter_bytes=Counter()
    for part in counter:
        b_tuple=tuple(bytes([b]) for b in part)
        counter_bytes[b_tuple]=counter[part]

    pair_counter=Counter()
    bytes_dict=defaultdict(list)
    for part,count in counter_bytes.items():
        for (x,y) in zip(part[:-1],part[1:]):
            pair_counter[(x,y)]+=count
            bytes_dict[(x,y)].append(part)
    
    for _ in range(iter_times):
        top_pair=max(pair_counter,key=lambda p:(pair_counter[p],p))
        new_byte=top_pair[0]+top_pair[1]
        vocab.append(new_byte)
        merge_rules.append(top_pair)
        merge(top_pair,new_byte,counter_bytes,pair_counter,bytes_dict)

    assert len(vocab)==vocab_size,"所得词汇表长度不符合要求"

    id_to_token={id:token for id,token in enumerate(vocab)}

    return (id_to_token,merge_rules)

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab=vocab
        self.re_vocab={b:id for id,b in vocab.items()}
        self.merges=merges
        self.merge_ranks={pair:i for i,pair in enumerate(merges)}
        self.special_tokens=special_tokens if special_tokens else []
        self._st_set=set(self.special_tokens)
        self._pat=regex.compile(PAT)
        if self.special_tokens:
            st_sorted=sorted(self.special_tokens,key=len,reverse=True)
            st_escaped=[regex.escape(p) for p in st_sorted]
            self._st_pattern="("+"|".join(st_escaped)+")"
        else:
            self._st_pattern=None

    def get_ids(
        self,
        match:bytes,
    )->list[int]:
        parts=[bytes([b]) for b in match]
        while len(parts)>1:
            best_rank=None
            best_i=None
            for i in range(len(parts)-1):
                rank=self.merge_ranks.get((parts[i],parts[i+1]))
                if rank is not None and (best_rank is None or rank<best_rank):
                    best_rank=rank
                    best_i=i
            if best_i is None:
                break
            merged=parts[best_i]+parts[best_i+1]
            parts=parts[:best_i]+[merged]+parts[best_i+2:]
        return [self.re_vocab[token] for token in parts]
    
    def encode(
        self,
        text:str
    )->list[int]:
        '''
        需要先对text进行pretokenize处理，把文本先切分成单独的pre_token
        pre_token进行encode，结果再按照merges转化为id_list
        '''
        text_list=self._pre_tokenize(text)
        st_set=self._st_set

        res_ids=[]
        if not text_list:
            return []
        for part in text_list:
            if not part:
                continue
            if part in st_set:
                res_ids.append(self.re_vocab[part.encode("utf-8",errors="ignore")])
            else:
                for m in self._pat.finditer(part):
                    match=m.group().encode("utf-8",errors="ignore")
                    res_ids.extend(self.get_ids(match))
        
        return res_ids

    def _pre_tokenize(self,text:str)->list[str]:
        if self._st_pattern is None:
            return [text]
        return regex.split(self._st_pattern,text)
        
    def encode_iterable(
        self,
        iterable:Iterable[str],
    )->Iterable[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(
        self,
        ids:list[int],
    )->str:
        return b''.join(self.vocab[id] for id in ids).decode("utf-8",errors="replace")
