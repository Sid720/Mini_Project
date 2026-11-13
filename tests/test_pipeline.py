import unittest
import pandas as pd
from pipeline import clean, features, snapshot, rank_metrics

class TemporalTests(unittest.TestCase):
    def setUp(self):
        raw = pd.DataFrame({'Customer ID':[1,1,1,2,2], 'Invoice':['100','101','102','103','C5'], 'InvoiceDate':['2011-01-01','2011-02-01','2011-03-03','2011-01-02','2011-01-03'], 'Quantity':[1]*5, 'Price':[10,20,1000,5,5], 'StockCode':['x']*5})
        self.df,self.audit=clean(raw)
    def test_future_does_not_change_features(self):
        a=features(self.df,'2011-02-01')
        b=features(self.df[self.df.date<'2011-02-01'],'2011-02-01')
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.loc['1','spend'],10)
    def test_label_boundaries(self):
        # Feb 1 is included; Mar 3 (30-day endpoint) is excluded.
        s=snapshot(self.df,'2011-02-01','2011-04-01').set_index('customer')
        self.assertEqual(s.loc['1','target'],1)
        only_end=self.df[self.df.invoice!='101']
        self.assertEqual(snapshot(only_end,'2011-02-01','2011-04-01').set_index('customer').loc['1','target'],0)
    def test_incomplete_labels_rejected(self):
        with self.assertRaises(ValueError): snapshot(self.df,'2011-02-01','2011-02-15')
    def test_cancellation_removed(self):
        self.assertEqual(self.audit['cancellations_nonpositive_removed'],1)
    def test_budget_metrics(self):
        m=rank_metrics([1,0,1,0],[.9,.2,.8,.1],.5)
        self.assertEqual(m['buyers_found'],2)
        self.assertEqual(m['lift'],2)
    def test_no_future_customers(self):
        self.assertTrue(features(self.df,'2010-01-01').empty)

if __name__=='__main__': unittest.main()
