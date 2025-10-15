package main

import (
	"fmt"

	rl "github.com/gen2brain/raylib-go/raylib"
)

type AIState int

const (
	Seeking    = 0
	Gathering  = 1
	Returning  = 2
	Patrolling = 3
)

type AppleThiefAI struct {
	Creature     *Creature
	State        AIState
	SightRange   float32
	TargetPos    rl.Vector2
	ScoreZone    *ScoreZone
	worldApples  *[]*Apple
	ChangedState bool
	Timer        float32
	TickCount    int
}

func NewAppleThiefAI(creature *Creature, scoreZone *ScoreZone, worldApples *[]*Apple) *AppleThiefAI {
	return &AppleThiefAI{
		Creature:     creature,
		State:        Seeking,
		SightRange:   1000,
		ScoreZone:    scoreZone,
		worldApples:  worldApples,
		ChangedState: false,
		Timer:        0,
		TickCount:    0,
	}
}

func (ai *AppleThiefAI) SetState(newState AIState) {
	ai.ChangedState = true
	ai.State = newState
}

func (ai *AppleThiefAI) Tick() {
	if ai.ChangedState {
		ai.Timer = 0
		ai.TickCount = 0
	} else {
		ai.Timer += rl.GetFrameTime()
		ai.TickCount++

	}
	switch ai.State {
	case Seeking:
		ai.TickSeek()
	case Gathering:
		ai.TickGather()
	case Returning:
		ai.TickReturn()
	case Patrolling:
		ai.TickPatrol()

	}
}

func (ai *AppleThiefAI) FindNearestApple() (*Apple, bool) {
	var nearestApple *Apple = nil
	minDist := float32(ai.SightRange)

	for _, apple := range *ai.worldApples {
		if apple.Carried {
			continue
		}
		dist := rl.Vector2Distance(ai.Creature.Pos, apple.Pos)
		if dist > ai.SightRange {
			continue
		}
		if dist < minDist {
			minDist = dist
			nearestApple = apple
		}
	}
	return nearestApple, nearestApple != nil
}

func (ai *AppleThiefAI) TickSeek() {
	if len(ai.Creature.Apples) >= CREATURE_MAX_APPLES {
		ai.SetState(Returning)
		return
	}

	if apple, found := ai.FindNearestApple(); found {
		ai.TargetPos = apple.Pos
		ai.SetState(Gathering)
	} else if len(ai.Creature.Apples) > 0 {
		ai.SetState(Returning)
	} else {
		ai.TargetPos = rl.NewVector2(
			ai.ScoreZone.Pos.X+float32(rl.GetRandomValue(-100, 100)),
			ai.ScoreZone.Pos.Y+float32(rl.GetRandomValue(-100, 100)),
		)
		ai.SetState(Patrolling)
	}
	ai.Creature.MoveCreatureTowards(ai.TargetPos)
}

func (ai *AppleThiefAI) TickGather() {
	dist := rl.Vector2Distance(ai.Creature.Pos, ai.TargetPos)

	if dist < APPLE_SIZE+CREATURE_SIZE {
		ai.Creature.GatherApples(ai.worldApples)
		ai.SetState(Seeking)
		ai.Creature.Stop()
		return
	}

	ai.Creature.MoveCreatureTowards(ai.TargetPos)
}

func (ai *AppleThiefAI) TickReturn() {
	if len(ai.Creature.Apples) == 0 {
		ai.SetState(Seeking)
		return
	}

	dist := rl.Vector2Distance(ai.Creature.Pos, ai.ScoreZone.Pos)

	if dist < SCORE_ZONE_SIZE {
		ai.Creature.DepositApple(ai.ScoreZone)
		if len(ai.Creature.Apples) == 0 {
			ai.SetState(Seeking)
		}
		ai.Creature.Stop()
		return
	}

	ai.Creature.MoveCreatureTowards(ai.ScoreZone.Pos)
}

func (ai *AppleThiefAI) TickRest() {
	if ai.TickCount == 0 {
		fmt.Println("I'm resting :3")
		ai.Creature.Stop()
	}

	if ai.Timer < 3 { //do nothing for 3 seconds
		return
	}
	ai.SetState(Seeking)
}

/*
Add a new State to the AI called Patrol.

If the AI can’t find any apples while seeking, it should switch to the patrol state.

The patrol state should have the AI select a random position near the Score Zone and move there.

Every 5 seconds, the AI should pick a new random position and move there instead.
*/

//this code sent me down a rabbit hole into agent-based AI
func (ai *AppleThiefAI) TickPatrol() {
	dist := rl.Vector2Distance(ai.Creature.Pos, ai.TargetPos)
    if (ai.TargetPos.X == 0 && ai.TargetPos.Y == 0) || dist < CREATURE_SIZE || ai.Timer >= 5 {
        fmt.Println("Patrolling")
        ai.TargetPos = rl.NewVector2(
            ai.ScoreZone.Pos.X+float32(rl.GetRandomValue(-100, 100)),
            ai.ScoreZone.Pos.Y+float32(rl.GetRandomValue(-100, 100)),
        )
        ai.Timer = 0
    }
    
    // Move toward target if we're not already close
    if dist >= CREATURE_SIZE {
        ai.Creature.MoveCreatureTowards(ai.TargetPos)
    } else {
        ai.Creature.Stop()
    }
}
