from agents.fql import FQLAgent
from agents.fbrac import FBRACAgent
from agents.rlpd import RLPDAgent
from agents.cgql import CGQLAgent
from agents.qam import QAMAgent
from agents.bam import BAMAgent
from agents.dsrl import DSRLAgent
from agents.dcgql import DCGQLAgent
from agents.fedit import FEditAgent
from agents.fawac import FAWACAgent
from agents.rebrac import ReBRACAgent
from agents.ifql import IFQLAgent
from agents.drift import DriftAgent
from agents.drift_affinity_q import DriftAffinityQAgent

agents = dict(
    ifql=IFQLAgent,
    fql=FQLAgent,
    fbrac=FBRACAgent,
    dsrl=DSRLAgent,
    qam=QAMAgent,
    bam=BAMAgent,
    fedit=FEditAgent,
    rlpd=RLPDAgent,
    cgql=CGQLAgent,
    fawac=FAWACAgent,
    rebrac=ReBRACAgent,
    dcgql=DCGQLAgent,
    drift=DriftAgent,
    drift_affinity_q=DriftAffinityQAgent,
)
