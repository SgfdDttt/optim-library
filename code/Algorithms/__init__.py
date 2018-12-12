import sys,os
sys.path.append(os.getcwd() + '/code/Algorithms')
# import all our algorithms
import oja
import msg_cca
#import minibatch_msg
#import l2_msg
#import rfmsg_cca

# left side of equal sign is the name that will be used to describe it in a config file
# right side of the equal sign is name_of_file.name_of_class
oja=oja.Oja


#import msg
#msg = msg.MSG

#import rfoja
#rfoja = rfoja.RFOja

#minibatch_msg = minibatch_msg.minibatchMSG
#l2_msg = l2_msg.l2MSG

msg_cca=msg_cca.MSG_CCA
#rfmsg_cca = rfmsg_cca.RFMSG_CCA

