"""Small CPU checks for clipping, masks, terminal shaping, and GAE."""
import torch
from task2_ppo.ppo import ppo_policy_loss, compute_gae, shaped_rewards


def main():
    ratios=torch.tensor([[1.5,0.5,1.5,0.5,100.]])
    advantage=torch.tensor([[1.,1.,-1.,-1.,999.]])
    mask=torch.tensor([[1.,1.,1.,1.,0.]])
    loss,_,fraction=ppo_policy_loss(ratios.log(),torch.zeros_like(ratios),advantage,mask,.2)
    torch.testing.assert_close(loss,torch.tensor(.15))
    torch.testing.assert_close(fraction,torch.tensor(1.))
    new=torch.tensor([[0.5]]).log().requires_grad_()
    loss,_,_=ppo_policy_loss(new,torch.zeros_like(new),torch.tensor([[-1.]]),torch.ones_like(new),.2)
    loss.backward();torch.testing.assert_close(new.grad,torch.zeros_like(new))
    mask=torch.tensor([[1.,1.,0.]])
    reward=shaped_rewards(torch.tensor([2.]),torch.tensor([[.3,.4,99.]]),
                         torch.tensor([[.1,.1,0.]]),mask,.1)
    torch.testing.assert_close(reward,torch.tensor([[-.02,1.97,0.]]))
    adv,returns=compute_gae(torch.tensor([[0.,1.,0.]]),torch.zeros(1,3),mask,1.,.95)
    torch.testing.assert_close(adv,torch.tensor([[.95,1.,0.]]))
    torch.testing.assert_close(returns,adv)
    print('PASS: clipping signs, clipping gradient, padding, terminal shaping, and GAE.')

if __name__=='__main__':main()
