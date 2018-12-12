import sys,os
sys.path.append(os.getcwd() + '/code/Algorithms')
# import all our algorithms
import oja
import msg_cca
import minibatch_msg

# left side of equal sign is the name that will be used to describe it in a config file
# right side of the equal sign is name_of_file.name_of_class
oja=oja.Oja


import msg
msg = msg.MSG

import rfoja
rfoja = rfoja.RFOja

minibatch_msg = minibatch_msg.minibatchMSG

msg_cca=msg_cca.MSG_CCA

