#include <iostream>

using namespace std;

int main(){
	int i,a,sum=0;
	cin>>i;
	for(int k=0; k<i; k++){
		cin>>a;
		sum+=a;
	}
	cout<<sum<<'\n';
	return 0;
}
