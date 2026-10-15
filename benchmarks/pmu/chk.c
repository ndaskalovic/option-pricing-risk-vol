#include <stdio.h>
int kc_check(int); int kc_num_groups(void); const char* kc_event_name(int,int); int kc_group_size(int);
int main(){ for(int g=0; g<kc_num_groups(); g++){ int m=kc_check(g); printf("group %d mask %d:", g, m); for(int i=0;i<kc_group_size(g);i++) printf(" %s%s", kc_event_name(g,i), (m>>i)&1?"":"(X)"); printf("\n"); } }
