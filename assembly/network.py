# -*- coding: utf-8 -*-
from torch import nn
from non_local_embedded_gaussian import NONLocalBlock1D
import torch
import torch.nn.functional as F


class CouplingBlocks(nn.Module):
    def __init__(self, args, recent_dim, in_dim_past):
        super(CouplingBlocks, self).__init__()

        self.dropout_rate = args.dropout_rate

        self.video_feat_dim = args.video_feat_dim
        self.latent_dim = args.latent_dim
        self.linear_dim = args.linear_dim

        self.recent_dim = recent_dim
        self.in_dim_past = in_dim_past
        self.past_attention = args.past_attention

        if self.past_attention:
            self.NLB_past = NONLocalBlock1D(args, self.in_dim_past, self.in_dim_past, self.latent_dim)
        self.NLB_recent = NONLocalBlock1D(args, self.recent_dim, self.in_dim_past, self.latent_dim)

        self.fc_recent = nn.Sequential(
            nn.Linear(in_features=2 * self.recent_dim * self.video_feat_dim, out_features=self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=self.linear_dim, out_features=self.linear_dim)
        )
        self.fc_context = nn.Sequential(
            nn.Linear(in_features=self.in_dim_past * self.video_feat_dim + 2 * self.recent_dim * self.video_feat_dim,
                      out_features=self.linear_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=self.linear_dim,
                      out_features=self.linear_dim)
        )

    def forward(self, spanning_snippets, recent_snippets):
        batch_size = spanning_snippets.size(0)

        if self.past_attention:
            nle_x_past = F.relu(self.NLB_past(spanning_snippets, spanning_snippets))
            nle_x_future = F.relu(self.NLB_recent(nle_x_past, recent_snippets))
            all_x_future = torch.cat((nle_x_future, recent_snippets), 1)
            all_x_task = torch.cat((nle_x_past, all_x_future), 1)
        else:
            nle_x_future = F.relu(self.NLB_recent(spanning_snippets, recent_snippets))
            all_x_future = torch.cat((nle_x_future, recent_snippets), 1)
            all_x_task = torch.cat((spanning_snippets, all_x_future), 1)

        output_future_fc = self.fc_recent(all_x_future.view(batch_size, -1))
        output_task_fc = self.fc_context(all_x_task.view(batch_size, -1))

        return output_future_fc, output_task_fc


class TemporalAggregateBlocks(nn.Module):
    def __init__(self, args):
        super(TemporalAggregateBlocks, self).__init__()

        self.linear_dim = args.linear_dim

        self.recent_dim = args.recent_dim
        self.span_dim1 = args.span_dim1
        self.span_dim2 = args.span_dim2
        self.span_dim3 = args.span_dim3

        self.CB1 = CouplingBlocks(args, self.recent_dim, self.span_dim1)
        self.CB2 = CouplingBlocks(args, self.recent_dim, self.span_dim2)
        self.CB3 = CouplingBlocks(args, self.recent_dim, self.span_dim3)

        self.fc_recent_tab = nn.Sequential(
            nn.Linear(in_features=3 * self.linear_dim, out_features=self.linear_dim)
        )

    def forward(self, spanning_snippets, recent_snippets):
        cb_recent1, cb_past1 = self.CB1(spanning_snippets[0], recent_snippets)
        cb_recent2, cb_past2 = self.CB2(spanning_snippets[1], recent_snippets)
        cb_recent3, cb_past3 = self.CB3(spanning_snippets[2], recent_snippets)

        cat_cb_recent = torch.cat((cb_recent1, cb_recent2, cb_recent3), 1)
        out_tab_recent = self.fc_recent_tab(cat_cb_recent)

        stack_cb_past = torch.stack((cb_past1, cb_past2, cb_past3), 0)
        out_tab_past = torch.max(stack_cb_past, 0)[0].squeeze(0)

        return out_tab_recent, out_tab_past


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        self.n_classes = args.action_class
        self.c_n_classes = args.coarse_class
        self.ts_n_classes = args.toyseq_class

        self.linear_dim = args.linear_dim
        self.single_latent = args.single_latent

        self.TAB1 = TemporalAggregateBlocks(args)
        self.TAB2 = TemporalAggregateBlocks(args)
        self.TAB3 = TemporalAggregateBlocks(args)
        self.TAB4 = TemporalAggregateBlocks(args)

        self.cls_act1 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=self.n_classes))
        self.cls_act2 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=self.n_classes))
        self.cls_act3 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=self.n_classes))
        self.cls_act4 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=self.n_classes))

        self.add_verb_loss = args.add_verb_loss
        self.add_object_loss = args.add_object_loss
        self.predict_latent = args.predict_latent
        self.predict_ts_latent = args.predict_ts_latent

        if args.add_verb_loss:
            self.cls_verb1 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=args.verb_class))
            self.cls_verb2 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=args.verb_class))
            self.cls_verb3 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=args.verb_class))
            self.cls_verb4 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=args.verb_class))

        if args.add_object_loss:
            self.cls_object1 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=args.object_class))
            self.cls_object2 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=args.object_class))
            self.cls_object3 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=args.object_class))
            self.cls_object4 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=args.object_class))

        if self.predict_latent:
            if not self.single_latent:
                self.lat_cls_act1 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=self.c_n_classes))
                self.lat_cls_act2 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=self.c_n_classes))
                self.lat_cls_act3 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=self.c_n_classes))
                self.lat_cls_act4 = nn.Sequential(nn.Linear(in_features=2 * self.linear_dim, out_features=self.c_n_classes))
            else:
                self.lat_cls_act1 = nn.Sequential(nn.Linear(in_features=4 * self.linear_dim, out_features=self.c_n_classes))

        if self.predict_ts_latent:
            self.ts_lat_cls_act1 = nn.Sequential(nn.Linear(in_features=4 * self.linear_dim, out_features=self.ts_n_classes))


    def forward(self, spanning_snippets, recent_snippets):
        #spanning_snippets = input[0]
        #recent_snippets = input[1]
        out_tab_recent1, out_tab_past1 = self.TAB1(spanning_snippets, recent_snippets[0])
        out_tab_recent2, out_tab_past2 = self.TAB2(spanning_snippets, recent_snippets[1])
        out_tab_recent3, out_tab_past3 = self.TAB3(spanning_snippets, recent_snippets[2])
        out_tab_recent4, out_tab_past4 = self.TAB4(spanning_snippets, recent_snippets[3])

        cat_tab1 = torch.cat((out_tab_recent1, out_tab_past1), 1)
        pred_act1 = self.cls_act1(cat_tab1)

        cat_tab2 = torch.cat((out_tab_recent2, out_tab_past2), 1)
        pred_act2 = self.cls_act2(cat_tab2)

        cat_tab3 = torch.cat((out_tab_recent3, out_tab_past3), 1)
        pred_act3 = self.cls_act3(cat_tab3)

        cat_tab4 = torch.cat((out_tab_recent4, out_tab_past4), 1)
        pred_act4 = self.cls_act4(cat_tab4)

        if self.add_verb_loss:
            pred_verb1 = self.cls_verb1(cat_tab1)
            pred_verb2 = self.cls_verb2(cat_tab2)
            pred_verb3 = self.cls_verb3(cat_tab3)
            pred_verb4 = self.cls_verb4(cat_tab4)
        else:
            pred_verb1 = None
            pred_verb2 = None
            pred_verb3 = None
            pred_verb4 = None

        if self.add_object_loss:
            pred_object1 = self.cls_object1(cat_tab1)
            pred_object2 = self.cls_object2(cat_tab2)
            pred_object3 = self.cls_object3(cat_tab3)
            pred_object4 = self.cls_object4(cat_tab4)
        else:
            pred_object1 = None
            pred_object2 = None
            pred_object3 = None
            pred_object4 = None

        # latent
        if self.predict_latent:
            if not self.single_latent:
                lat_pred_act1 = self.lat_cls_act1(cat_tab1)
                lat_pred_act2 = self.lat_cls_act2(cat_tab2)
                lat_pred_act3 = self.lat_cls_act3(cat_tab3)
                lat_pred_act4 = self.lat_cls_act4(cat_tab4)
            else:
                out_tab_past = torch.cat((out_tab_past1, out_tab_past2,
                                          out_tab_past3, out_tab_past4), dim=1)
                lat_pred_act1 = self.lat_cls_act1(out_tab_past)
                lat_pred_act2 = None
                lat_pred_act3 = None
                lat_pred_act4 = None
        else:
            lat_pred_act1 = None
            lat_pred_act2 = None
            lat_pred_act3 = None
            lat_pred_act4 = None


        # toy seq latent
        if self.predict_ts_latent:
            out_tab_past = torch.cat((out_tab_past1, out_tab_past2,
                                      out_tab_past3, out_tab_past4), dim=1)
            ts_lat_pred_act1 = self.ts_lat_cls_act1(out_tab_past)
        else:
            ts_lat_pred_act1 = None
           

        return pred_act1, pred_act2, pred_act3, pred_act4, \
               pred_verb1, pred_verb2, pred_verb3, pred_verb4, \
               pred_object1, pred_object2, pred_object3, pred_object4, \
               lat_pred_act1, lat_pred_act2, lat_pred_act3, lat_pred_act4, \
               ts_lat_pred_act1


