"""Small CPU tests of grouping, sign-aware clipping, masks and gradient allocation."""
import torch
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

def main():
    rewards = torch.tensor([1.,3.,100.,104.,7.,7.])
    ids = torch.tensor([8,8,2,2,5,5])
    adv = group_relative_advantages(rewards, ids)
    torch.testing.assert_close(adv,torch.tensor([-1.,1.,-1.,1.,0.,0.]),atol=2e-6,rtol=0)
    shifted=rewards+torch.where(ids==2,500.,0.)
    torch.testing.assert_close(adv,group_relative_advantages(shifted,ids))
    ratio=torch.tensor([[1.5,.5,1.5,.5]])
    advantage=torch.tensor([1.,1.,-1.,-1.])
    for r,a,want in zip(ratio[0],advantage,[-1.2,-.5,1.5,.8]):
        lp=r.log().reshape(1,1).requires_grad_()
        loss,_=grpo_policy_loss(lp,torch.zeros_like(lp),a.reshape(1),torch.ones_like(lp),torch.zeros_like(lp),.2,0.)
        torch.testing.assert_close(loss,torch.tensor(want))
    mask=torch.tensor([[1.,1.,0.,0.],[1.,1.,1.,1.],[0.,0.,0.,0.]])
    for loss_type,expected in [('grpo',torch.tensor([[1/6,1/6,0.,0.],[1/12]*4,[0.]*4])),
                               ('dr_grpo',torch.tensor([[1/12,1/12,0.,0.],[1/12]*4,[0.]*4]))]:
        lp=torch.zeros_like(mask,requires_grad=True)
        loss,_=grpo_policy_loss(lp,lp.detach(),torch.ones(3),mask,lp.detach(),.2,0.,loss_type,4)
        loss.backward();torch.testing.assert_close(lp.grad.abs(),expected)
    lp=torch.zeros_like(mask,requires_grad=True)
    loss,_=grpo_policy_loss(lp,lp.detach(),torch.ones(3),torch.zeros_like(mask),lp.detach(),.2,.1)
    loss.backward();assert loss.item()==0 and lp.grad.abs().sum().item()==0
    torch.testing.assert_close(mask_truncated_sequences(mask,[False,True,False]),torch.tensor([[1.,1.,0.,0.],[0.]*4,[0.]*4]))
    # Accumulating sequence microbatches must preserve the full-batch objective.
    for kind in ['grpo','dr_grpo']:
        initial=torch.tensor([[.1,-.1,.2,0.],[.3,0.,-.2,.1],[0.]*4])
        whole=initial.clone().requires_grad_();micro=initial.clone().requires_grad_()
        old=torch.zeros_like(mask);reference=torch.full_like(mask,-.1)
        advantages=torch.tensor([1.,-.5,0.])
        full,_=grpo_policy_loss(whole,old,advantages,mask,reference,.2,.1,kind,4)
        full.backward();total=0.
        for i in range(3):
            n=mask[i].sum().item()
            if not n:continue
            local,_=grpo_policy_loss(micro[i:i+1],old[i:i+1],advantages[i:i+1],mask[i:i+1],
                reference[i:i+1],.2,.1*3*n/mask.sum().item(),kind,4)
            total=total+local/3
        total.backward()
        torch.testing.assert_close(total,full)
        torch.testing.assert_close(micro.grad,whole.grad)
    print('PASS: within-prompt grouping, constant groups, PPO clipping signs, normalization gradients, and truncation masks.')

if __name__ == '__main__':main()
