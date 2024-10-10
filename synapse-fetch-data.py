# Fetch BraTS data from synapse, use synapse config to add auth token

import synapseclient
import synapseutils

syn = synapseclient.Synapse()
syn.login()


syn.get("syn59407686", downloadLocation="data/")
syn.get("syn59860022", downloadLocation="data/")
syn.get("syn61596964", downloadLocation="data/")

syn.get("syn61453479", downloadLocation="data/")
