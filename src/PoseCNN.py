# ====================CLASS PoseCNN========================
# This part of network structure is the appended structrue 
# that takes inputs from 2 separate barnch of GoogLeNet and 
# generate rotation and parameters at the end.
# 
# Three inputs from previous part of networks are: 
# "1st Auxiliary Brach output"
# "2nd Auxiliary Brach output"
# "Main branch output"
#
# The mixed output obtained from two branches are then 
# concatenated and feed into translation and rotation branches.
#
# Two outputs are:
# "Translation parameter x3"
# "Rotation parameter x4"
# =============================================================

from network import Network

class PoseCNN(Network):
    def setup(self):
        # ======================Main Brach=======================
        # From (intermaiate) Main branch input 7x7x1024
        # To pool1 (output) 4x4x1024
        # ======================Main Brach=======================
        (self.feed('main_branch')
             #.conv(3, 3, 1792, 2, 2, name='pool1'))
             .max_pool(2, 2, 2, 2, name='pool1'))
        
        # ======================Concatenate=======================
        # From 2x intermaiate Auxiliary branch input 4x4x256
        # and pool1 4x4x1024
        # To concat_out 4x4x1280
        # ======================Concatenate=======================
        (self.feed('pool1', 
                   'aux1_branch', 
                   'aux2_branch')
             .concat(3, name='concat_out'))
        
        # ======================Rotation Branch=======================
        # From concate_out 4x4x1280
        # To rot_out 1x4
        # ======================Rotation Branch=======================
        (self.feed('concat_out')
             .conv(2, 2, 2048, 2, 2, padding='VALID', name='rot_conv')
             .avg_pool(2, 2, 1, 1, padding='VALID', name='rot_pool')
             .fc(2048, name='rot_fc')
             .fc(4, relu=False, name='rot_out'))
        
        # ======================translation Branch=====================
        # From concate_out 4x4x1280
        # To trans_out 1x3
        # ======================Translation Branch=====================
        (self.feed('concat_out')
             .conv(2, 2, 2048, 2, 2, padding='VALID', name='trans_conv')
             .avg_pool(2, 2, 1, 1, padding='VALID', name='trans_pool')
             .fc(2048, name='trans_fc')
             .fc(3, relu=False, name='trans_out'))