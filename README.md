# 動機
[Shalit et al. (2017)](https://arxiv.org/pdf/1606.03976.pdf)を自力で実装してみる。

# setup
At first, add the following in your `~/.bashrc`.
```
function conda_cd() {
    cd $@ && [ -f ".workspace" ] && source .workspace
}
alias cd="conda_cd"
```
