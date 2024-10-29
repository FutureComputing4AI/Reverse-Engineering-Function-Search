from sys import path
path.insert(1, '../')

import numpy as np

from evaluation_modules import compute_mrr, mask_types, normalize_labels

def test_compute_mrr():
    labels = np.array([0,0,1,1,2,2])
    neighbors = np.array([[1, 2], [0, 2],
                             [4, 5], [4, 5],
                             [0, 5], [0, 4]])
    ks = [1,2] 
    upper, lower, recalls = compute_mrr(labels, neighbors, ks=ks)
    assert upper.item()==(22/36)
    assert lower.item()== (0.5)
    assert len(recalls)==len(ks)
    assert recalls[1]==(1/3)
    assert recalls[2]==(2/3)

class LabelsDataset():
    def __init__(self, mapping):
        self.mapping = mapping
    def get_name(self, label):
        return self.mapping[label]

def test_normalize_labels():
    labels_mapping = {1: '1\\a',
               2: '2\\a',
               3: '3\\a', 
               4: '1\\b',
               5: '2\\b', 
               6: '1\\c',
               7: '1\\d',
               8: '4\\e'}

    dataset = LabelsDataset(labels_mapping)
    
    labels_1 = np.array([1, 2, 3, 4, 5, 6, 1, 2, 5, 6])
    how = 'source'

    new_labels_1, mapping_1 = normalize_labels(labels_1, dataset, how)

    true_mapping_1 = {'a': 0, 'b': 1, 'c': 2}

    assert (new_labels_1==np.array([0,0,0,1,1,2,0,0,1,2])).all()
    assert mapping_1==true_mapping_1

    labels_2 = np.array([3, 3, 4, 7, 8])

    new_labels_2, mapping_2 = normalize_labels(labels_2, dataset, how)

    true_mapping_2 = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e':4}

    assert (new_labels_2==np.array([0,0,1,3,4])).all()
    assert mapping_2 == true_mapping_2

            
def test_mask_types():
    " function names are weird; I will try to capture as many edge cases as possible here"

    # paths 1, 2, 7, 9, 14, 15, 18, 19, 21, 23 
    str1_in = 'std::forward<<lambda_98edac8c665df57bd5e22d3f7e537fee>const&__ptr64>'
    str1_out = 'std::forward<<lambda_98edac8c665df57bd5e22d3f7e537fee>#>'

    # paths 1, 3, 4, 5, 6, 9, 14, 16, 17, 18, 19, 23 
    str2_in = 'std::basic_ostream<char,std::char_traits<char>>::operator<<'
    str2_out = 'std::basic_ostream<#,#<#>>::operator<<'

    # paths 1, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 22 
    str3_in = 'std::_Tree_iterator<std::_Tree_val<std::_Tree_simple_types<std::pair<std::\
               basic_string<char,std::char_traits<char>,std::allocator<char>>const,int>>>>::operator->'
    str3_out = 'std::_Tree_iterator<#<#<#<#<#,#<#>,#<#>>#,#>>>>::operator->'
    str3_out_alt = 'std::_Tree_iterator<$<$<$<$<$,$<$>,$<$>>$,$>>>>::operator->'

    # paths 1, 2, 7, 8, 9, 14, 16, 17, 18, 19, 23, 24 
    str4_in = 'std::_Accumulate_unchecked<charconst*__ptr64,unsigned__int64,<lambda_3f3094287d34f5112b58895a7d439ffb>>'
    str4_out = 'std::_Accumulate_unchecked<#,#,<lambda_3f3094287d34f5112b58895a7d439ffb>>'

    # paths 1, 2, 3, 5, 6, 9, 14, 15, 16, 17, 18, 19, 20, 23
    str5_in = 'std::_Compressed_pair<std::default_delete<std::tuple<<lambda_baf9e60028b58d7c0b4327fcc208a217>,std::\
                unordered_map<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,std::set<std::\
                basic_string<char,std::char_traits<char>,std::allocator<char>>,std::less<std::basic_string<char,\
                std::char_traits<char>,std::allocator<char>>>,std::allocator<std::basic_string<char,std::\
                char_traits<char>,std::allocator<char>>>>,std::hash<std::basic_string<char,std::char_traits<char>,\
                std::allocator<char>>>,std::equal_to<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>,\
                std::allocator<std::pair<std::basic_string<char,std::char_traits<char>,std::allocator<char>>const,\
                std::set<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,std::less<std::\
                basic_string<char,std::char_traits<char>,std::allocator<char>>>,std::allocator<std::basic_string<char,std::\
                char_traits<char>,std::allocator<char>>>>>>>>>,std::tuple<<lambda_baf9e60028b58d7c0b4327fcc208a217>,std::\
                unordered_map<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,std::\
                set<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,std::less<std::basic_string<char,\
                std::char_traits<char>,std::allocator<char>>>,std::allocator<std::basic_string<char,std::char_traits<char>,\
                std::allocator<char>>>>,std::hash<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>,\
                std::equal_to<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>,std::allocator<std::\
                pair<std::basic_string<char,std::char_traits<char>,std::allocator<char>>const,std::set<std::\
                basic_string<char,std::char_traits<char>,std::allocator<char>>,std::less<std::basic_string<char,\
                std::char_traits<char>,std::allocator<char>>>,std::allocator<std::basic_string<char,std::char_traits<char>,\
                std::allocator<char>>>>>>>>*,1>::_Compressed_pair<std::default_delete<std::\
                tuple<<lambda_baf9e60028b58d7c0b4327fcc208a217>,std::unordered_map<std::basic_string<char,\
                std::char_traits<char>,std::allocator<char>>,std::set<std::basic_string<char,std::char_traits<char>,\
                std::allocator<char>>,std::less<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>,\
                std::allocator<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>>,\
                std::hash<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>,std::equal_to<std::\
                basic_string<char,std::char_traits<char>,std::allocator<char>>>,std::allocator<std::pair<std::\
                basic_string<char,std::char_traits<char>,std::allocator<char>>const,std::set<std::basic_string<char,\
                std::char_traits<char>,std::allocator<char>>,std::less<std::basic_string<char,std::char_traits<char>,\
                std::allocator<char>>>,std::allocator<std::basic_string<char,std::char_traits<char>,\
                std::allocator<char>>>>>>>>>,std::tuple<<lambda_baf9e60028b58d7c0b4327fcc208a217>,\
                std::unordered_map<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,\
                std::set<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,std::less<std::\
                basic_string<char,std::char_traits<char>,std::allocator<char>>>,std::allocator<std::basic_string<char,\
                std::char_traits<char>,std::allocator<char>>>>,std::hash<std::basic_string<char,std::char_traits<char>,\
                std::allocator<char>>>,std::equal_to<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>,\
                std::allocator<std::pair<std::basic_string<char,std::char_traits<char>,std::allocator<char>>const,\
                std::set<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,std::less<std::\
                basic_string<char,std::char_traits<char>,std::allocator<char>>>,std::allocator<std::basic_string<char,\
                std::char_traits<char>,std::allocator<char>>>>>>>>*,1><std::tuple<<lambda_baf9e60028b58d7c0b4327fcc208a217>,\
                std::unordered_map<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,\
                std::set<std::basic_string<char,std::char_traits<char>,std::allocator<char>>,std::less<std::basic_string<char,\
                std::char_traits<char>,std::allocator<char>>>,std::allocator<std::basic_string<char,std::char_traits<char>,\
                std::allocator<char>>>>,std::'
    
    str5_out = 'std::_Compressed_pair<#<#<<lambda_baf9e60028b58d7c0b4327fcc208a217>,#<#<#,#<#>,#<#>>,#<#<#,#<#>,#<#>>,\
#<#<#,#<#>,#<#>>>,#<#<#,#<#>,#<#>>>>,#<#<#,#<#>,#<#>>>,#<#<#,#<#>,#<#>>>,#<#<#<#,#<#>,#<#>>#,#<#<#,#<#>,#<#>>,#<#<#,#<#>,\
#<#>>>,#<#<#,#<#>,#<#>>>>>>>>>,#<<lambda_baf9e60028b58d7c0b4327fcc208a217>,#<#<#,#<#>,#<#>>,#<#<#,#<#>,#<#>>,#<#<#,#<#>,\
#<#>>>,#<#<#,#<#>,#<#>>>>,#<#<#,#<#>,#<#>>>,#<#<#,#<#>,#<#>>>,#<#<#<#,#<#>,#<#>>#,#<#<#,#<#>,#<#>>,#<#<#,#<#>,#<#>>>,\
#<#<#,#<#>,#<#>>>>>>>>#,#>::_Compressed_pair<#<#<<lambda_baf9e60028b58d7c0b4327fcc208a217>,#<#<#,#<#>,#<#>>,#<#<#,#<#>,\
#<#>>,#<#<#,#<#>,#<#>>>,#<#<#,#<#>,#<#>>>>,#<#<#,#<#>,#<#>>>,#<#<#,#<#>,#<#>>>,#<#<#<#,#<#>,#<#>>#,#<#<#,#<#>,#<#>>,\
#<#<#,#<#>,#<#>>>,#<#<#,#<#>,#<#>>>>>>>>>,#<<lambda_baf9e60028b58d7c0b4327fcc208a217>,#<#<#,#<#>,#<#>>,#<#<#,#<#>,#<#>>,\
#<#<#,#<#>,#<#>>>,#<#<#,#<#>,#<#>>>>,#<#<#,#<#>,#<#>>>,#<#<#,#<#>,#<#>>>,#<#<#<#,#<#>,#<#>>#,#<#<#,#<#>,#<#>>,#<#<#,#<#>,\
#<#>>>,#<#<#,#<#>,#<#>>>>>>>>#,#><#<<lambda_baf9e60028b58d7c0b4327fcc208a217>,#<#<#,#<#>,#<#>>,#<#<#,#<#>,#<#>>,#<#<#,#<#>,\
#<#>>>,#<#<#,#<#>,#<#>>>>,#'

    str6_in_out = 'PaintDLL::Circle::Draw'

    # NOTE: KNOWN FAILURE CASE FOR THE FUNCTION NAME '>>'. The error is caught by the exception handler is and the original function name,
    # '>>', is returned. This seems like a reasonable output for this input.
    str7_in_out = '>>'

    # ANOTHER KNOWN FAILURE CASE IS '<<', which returns '<<#', but should probably just return '<<'.

    assert mask_types(str1_in)==str1_out
    assert mask_types(str2_in)==str2_out
    assert mask_types(str3_in)==str3_out
    assert mask_types(str4_in)==str4_out
    assert mask_types(str5_in)==str5_out
    assert mask_types(str6_in_out)==str6_in_out
    assert mask_types(str7_in_out)==str7_in_out
    assert mask_types(str3_in, '$')==str3_out_alt

tests = [test_compute_mrr, test_mask_types, test_normalize_labels]

if __name__=='__main__':
    for test in tests:
        test()

    print("All tests passed!")
