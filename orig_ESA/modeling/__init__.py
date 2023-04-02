
from .modeling_speakergraph import ElectraForQuestionAnswering as ElectraForSGCN
from .modeling_speakergraph import BertForQuestionAnswering as BertForSGCN
from .modeling_speakergraph import RobertaForQuestionAnswering as RobertaForSGCN
from .modeling_discoursegraph import ElectraForQuestionAnswering as ElectraForGCN
from .modeling_discoursegraph import BertForQuestionAnswering as BertForGCN
from .modeling_discoursegraph import RobertaForQuestionAnswering as RobertaForGCN
# from .modeling_final import BertForQuestionAnswering as BertForGCN1
# from .modeling_final import ElectraForQuestionAnswering as ElectraForGCN1
# from .modeling_final import RobertaForQuestionAnswering as RobertaForGCN1
from .modeling_all import BertForQuestionAnswering as Bertall
# from .modeling_all_com import BertForQuestionAnswering as Bertall
from .modeling_all import ElectraForQuestionAnswering as Electraall
from .modeling_wo_mdfn import BertForQuestionAnswering as Bertwomdfn
from .modeling_wo_dg import BertForQuestionAnswering as Bertwodg
from .modeling_wo_sg import BertForQuestionAnswering as Bertwosg
from .modeling_all import ElectraForQuestionAnswering31  as Electra31
from .modeling_all import ElectraForQuestionAnswering21  as Electra21
from .modeling_all import ElectraForQuestionAnswering33  as Electra33